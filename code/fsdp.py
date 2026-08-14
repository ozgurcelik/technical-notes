from __future__ import annotations

import torch
import torch.distributed as dist

from dataclasses import dataclass, field


@dataclass
class ShardMetadata:
    num_elements: int
    shard_size: int
    padded_num_elements: int
    shape: torch.Size
    start: int
    end: int

@dataclass
class FSDPParamState:
    name: str
    metadata: ShardMetadata
    local_param: torch.nn.Parameter
    full_param: torch.nn.Parameter | None = None
    forward_gather_handle: dist.Work | None = None
    backward_gather_handle: dist.Work | None = None

@dataclass
class FSDPLayerState:
    layer: torch.nn.Module
    param_states: dict[str, FSDPParamState] = field(default_factory=dict)

@dataclass
class PendingReduceScatter:
    handle: dist.Work | None  # async Work handle (None when world_size == 1)
    local_grad: torch.Tensor  # the local shard grad; valid after wait()
    local_param: torch.nn.Parameter | None = None  # receives the finalized grad

def _cast_floating(obj, dtype: torch.dtype):
    """
    Cast every floating-point Tensor in obj (Tensor / tuple / list / dict) to
    `dtype` via a differentiable .to(); leave integer/bool tensors (e.g.
    Embedding indices) and non-tensors untouched.
    """
    if torch.is_tensor(obj):
        return obj.to(dtype) if obj.is_floating_point() else obj
    if isinstance(obj, tuple):
        return tuple(_cast_floating(o, dtype) for o in obj)
    if isinstance(obj, list):
        return [_cast_floating(o, dtype) for o in obj]
    if isinstance(obj, dict):
        return {key: _cast_floating(value, dtype) for key, value in obj.items()}
    return obj

class FSDP(torch.nn.Module):
    def __init__(self, 
                 module: torch.nn.Module,
                 compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.module = module
        self.compute_dtype = compute_dtype

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self._prefetch_window_size = 2
        self._reduce_scatter_window_size = 2

        self._broadcast_initial_state()

        self._layer_states: dict[torch.nn.Module, FSDPLayerState] = {}
        self._forward_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._backward_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._pending_reduce_scatters: list[PendingReduceScatter] = []

        self._fsdp_layers: list[torch.nn.Module] = []
        self._replicate_parameters: list[torch.nn.Parameter] = []
        self._find_fsdp_layers_and_replicated_parameters()
        self._cast_replicated_params_to_float32()
        self._layer_index = {layer: i for i, layer in enumerate(self._fsdp_layers)}
        self._create_layer_states()
        self._register_forward_hooks()
        self._register_backward_hooks()

    def _broadcast_initial_state(self) -> None:
        """
        Broadcast the initial state of the model to all the ranks.
        """
        if self.world_size == 1:
            return
        
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0)
            
    def _find_fsdp_layers_and_replicated_parameters(self) -> None:
        """
        Find all the layers that are going to be sharded.
        Also find all the parameters that are going to be replicated.
        """
        for submodule in self.module.modules():
            if isinstance(submodule, (torch.nn.Linear, torch.nn.Embedding)):
                self._fsdp_layers.append(submodule)
            else:
                for param in submodule.parameters(recurse=False):
                    if param.requires_grad:
                        self._replicate_parameters.append(param)

    def _cast_replicated_params_to_float32(self) -> None:
        """
        Cast the replicated parameters to float32.
        """
        for param in self._replicate_parameters:
            param.data = param.data.to(torch.float32)
    
    def _get_shard_metadata(self, param: torch.nn.Parameter) -> ShardMetadata:
        """
        Get the metadata for the shard.
        This will be used to get the local shard, reconstruct the full parameter, and all-gather the parameter.
        """
        num_elements = param.numel()
        shard_size = (num_elements + self.world_size - 1) // self.world_size
        padded_num_elements = shard_size * self.world_size
        start = self.rank * shard_size
        end = start + shard_size
        return ShardMetadata(
            num_elements=num_elements,
            shard_size=shard_size,
            padded_num_elements=padded_num_elements,
            shape=param.shape,
            start=start,
            end=end,
        )

    def _get_local_shard_and_metadata(self, param: torch.nn.Parameter) -> tuple[torch.Tensor, ShardMetadata]:
        """
        Get the local shard of the parameter and the metadata.
        """
        metadata = self._get_shard_metadata(param)

        flattened_param = param.detach().flatten()

        if metadata.padded_num_elements > metadata.num_elements:
            padding = torch.zeros(metadata.padded_num_elements - metadata.num_elements, dtype=param.dtype, device=param.device)
            flattened_param = torch.cat([flattened_param, padding])

        # Master weight is always in float32.
        local_shard = flattened_param[metadata.start:metadata.end].clone().to(torch.float32)
        return local_shard, metadata

    def _create_layer_state(self, layer: torch.nn.Module) -> FSDPLayerState:
        """
        Create the state for the layer and set the local parameter to the layer.
        """
        param_states: dict[str, FSDPParamState] = {}
        for param_name in ("weight", "bias"):
            param = getattr(layer, param_name, None)
            if param is None:
                continue

            # inherit the requires_grad from the original parameter
            local_shard, metadata = self._get_local_shard_and_metadata(param)
            param_state = FSDPParamState(
                name=param_name,
                metadata=metadata,
                local_param=torch.nn.Parameter(local_shard, requires_grad=param.requires_grad),
            )
            param_states[param_name] = param_state
            setattr(layer, param_name, param_state.local_param)
            
        return FSDPLayerState(
            layer=layer,
            param_states=param_states,
        )

    def _create_layer_states(self) -> None:
        """
        Create the states for the FSDP layers.
        """
        for layer in self._fsdp_layers:
            self._layer_states[layer] = self._create_layer_state(layer)

    def _all_gather_param_async(self, local_param: torch.nn.Parameter, 
                            metadata: ShardMetadata) -> tuple[dist.Work | None, torch.Tensor]:
        """
        Issue an async all-gather operation.
        """
        local_shard = local_param.detach()
        if self.compute_dtype is not None:
            local_shard = local_shard.to(self.compute_dtype)
        if self.world_size == 1:
            return None, local_shard.clone()
        buffer = torch.empty(metadata.padded_num_elements, dtype=local_shard.dtype, device=local_shard.device)
        handle = dist.all_gather_into_tensor(buffer, local_shard, async_op=True)
        return handle, buffer

    # Full-parameter storage lifecycle. Raw storage operations are deliberately
    # confined to these helpers because autograd may retain aliases of the
    # gathered parameter between forward and backward.
    @staticmethod
    def _full_param_storage_is_allocated(param: torch.nn.Parameter | None) -> bool:
        """A full param's memory is tracked by its STORAGE, not its tensor
        sizes: we free by resizing the storage to 0 while keeping the [out, in]
        sizes (so autograd still sees the right shape). So 'is it materialized?'
        must ask the storage, not `.data.numel()`."""
        return param is not None and param.untyped_storage().size() > 0

    @staticmethod
    def _release_full_param_storage(full_param: torch.nn.Parameter | None) -> None:
        """Release a full (unsharded) weight by resizing its storage to 0.

        Autograd's saved-for-backward copy of the weight shares this exact
        storage, so shrinking it releases the shared allocation for reuse by the
        allocator. Assigning ``data = empty(0)`` does not: it points the parameter
        at a fresh empty storage while the saved copy keeps the old allocation
        alive. On CUDA, releasing a live allocation for reuse does not necessarily
        reduce the amount of memory reserved by the caching allocator. The tensor
        keeps its [out, in] sizes so AccumulateGrad still accepts the full-shaped
        gradient."""
        if full_param is None:
            return
        with torch.no_grad():
            full_param.untyped_storage().resize_(0)

    def _rematerialize_full_param_storage_async(
        self,
        full_param: torch.nn.Parameter,
        local_param: torch.nn.Parameter,
        metadata: ShardMetadata,
    ) -> dist.Work | None:
        """Re-materialize the full weight IN PLACE for backward by refilling the
        same storage the forward pass freed. Because autograd's saved copy shares
        that storage, this is what makes the saved weight valid again (it's the
        backward ALL-GATHER box in the FSDP diagram), rather than allocating a
        second, unused copy."""
        local_shard = local_param.detach()
        if self.compute_dtype is not None:
            local_shard = local_shard.to(self.compute_dtype)

        with torch.no_grad():
            storage = full_param.untyped_storage()
            storage.resize_(metadata.padded_num_elements * full_param.element_size())
            flat = torch.empty(0, dtype=full_param.dtype, device=full_param.device)
            flat.set_(storage, 0, (metadata.padded_num_elements,))
            full_param.data = flat

        if self.world_size == 1:
            with torch.no_grad():
                full_param.data.copy_(local_shard)
            return None
        return dist.all_gather_into_tensor(full_param.data, local_shard, async_op=True)

    def _prefetch_layer_backward(self, layer: torch.nn.Module) -> None:
        """
        Issue an async all-gather operation for the layers params for the backward pass.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            full_param = param_state.full_param
            if full_param is None or self._full_param_storage_is_allocated(full_param):
                continue
            metadata = param_state.metadata
            if metadata is None:
                continue
            handle = self._rematerialize_full_param_storage_async(
                full_param, param_state.local_param, metadata
            )
            param_state.backward_gather_handle = handle

    def _reduce_scatter_grad_async(self, full_grad: torch.Tensor, metadata: ShardMetadata) -> PendingReduceScatter:
        """
        We issue an async reduce scatter operation.
        """
        flattened_grad = full_grad.flatten()
        if metadata.padded_num_elements > metadata.num_elements:
            padding = torch.zeros(metadata.padded_num_elements - metadata.num_elements, dtype=full_grad.dtype, device=full_grad.device)
            flattened_grad = torch.cat([flattened_grad, padding])

        if self.world_size == 1:
            return PendingReduceScatter(
                handle=None,
                local_grad=flattened_grad,
                local_param=None,
            )
        else:
            local_grad = torch.empty(metadata.shard_size, dtype=full_grad.dtype, device=full_grad.device)
            handle = dist.reduce_scatter_tensor(output=local_grad, input=flattened_grad, op=dist.ReduceOp.SUM, async_op=True)
            return PendingReduceScatter(
                handle=handle,
                local_grad=local_grad,
                local_param=None,
            )

    def _finalize_reduce_scatter(self, pending: PendingReduceScatter) -> None:
        if pending.handle is not None:
            pending.handle.wait()

        local_grad = pending.local_grad.to(pending.local_param.dtype).div_(self.world_size)

        if pending.local_param.grad is None:
            pending.local_param.grad = local_grad
        else:
            pending.local_param.grad.add_(local_grad)

    def _drain_reduce_scatters(self) -> None:
        """
        Drain the reduce scatters.
        """
        while len(self._pending_reduce_scatters) > self._reduce_scatter_window_size:
            pending = self._pending_reduce_scatters.pop(0)
            self._finalize_reduce_scatter(pending)

    def _make_reduce_scatter_hook(self, layer: torch.nn.Module, param_name: str):
        """
        Return a post-accumulate-grad hook that reduce scatters the full
        weight's gradient into the local shard, then frees the full weight.
        """
        def hook(param: torch.nn.Parameter):
            param_state = self._layer_states[layer].param_states[param_name]
            if param.grad is not None:
                pending_reduce_scatter = self._reduce_scatter_grad_async(param.grad, param_state.metadata)
                # share the same object (a reference, not a copy)
                pending_reduce_scatter.local_param = param_state.local_param
                self._pending_reduce_scatters.append(pending_reduce_scatter)
                self._drain_reduce_scatters()
            
            param.grad = None
            # Free the full weight now that this layer's backward has consumed
            # it: resize the storage to 0 (the autograd-saved copy shares it, so
            # the shared allocation is released for reuse) rather than detaching
            # to a new empty storage.
            self._release_full_param_storage(param)
        return hook

    def _prefetch_layer_forward(self, layer: torch.nn.Module) -> None:
        """
        Issue an async all-gather operation for the layers params for the forward pass.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if self._full_param_storage_is_allocated(param_state.full_param):
                continue
            local_param = getattr(layer, param_name, None)
            metadata = param_state.metadata
            if local_param is None or metadata is None:
                continue
            handle, buffer = self._all_gather_param_async(local_param, metadata)
            # inherit the requires_grad from the original parameter and add the post-accumulate-grad hook if it requires grad
            # we add the hook here because hook needs to be attached to the full param before the forward pass.
            # if we were to wait till the prefetch backward pass, then the autograd would have already built the graph from forward pass
            # the hooks need to be attached to the object that participates in the forward pass.
            param_state.full_param = torch.nn.Parameter(buffer, requires_grad=local_param.requires_grad)
            if param_state.full_param.requires_grad:
                param_state.full_param.register_post_accumulate_grad_hook(self._make_reduce_scatter_hook(layer, param_name))
            param_state.forward_gather_handle = handle

    def _use_prefetched_layer_forward(self, layer: torch.nn.Module) -> None:
        """
        Wait for the all-gathers, trim the padding, reshape the parameter, and attach it to the layer.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if param_state.forward_gather_handle is not None:
                param_state.forward_gather_handle.wait()
            full_param = param_state.full_param
            full_param.data = full_param.data[:param_state.metadata.num_elements].view(param_state.metadata.shape)
            setattr(layer, param_name, full_param)
            param_state.forward_gather_handle = None

    def _use_prefetched_layer_backward(self, layer: torch.nn.Module) -> None:
        """
        Wait for the all-gathers, trim the padding, reshape the parameter, and attach it to the layer.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            if param_state.backward_gather_handle is not None:
                param_state.backward_gather_handle.wait()
            param_state.full_param.data = param_state.full_param.data[:param_state.metadata.num_elements].view(param_state.metadata.shape)
            param_state.backward_gather_handle = None

    def _pre_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        """
        Pre-forward hook for the FSDP layers.
        Wait for the prefetched parameters before the layer's forward pass.
        """
        self._prefetch_layer_forward(layer) # if the layer is already prefetched, this is a no-op
        self._use_prefetched_layer_forward(layer)

    def _post_forward_hook(self, layer: torch.nn.Module, inputs: tuple[torch.Tensor, ...], outputs: torch.Tensor):
        """
        We should restore the local parameters after the forward pass to the layer.
        And clear the full parameter data.
        """
        for param_name, param_state in self._layer_states[layer].param_states.items():
            setattr(layer, param_name, param_state.local_param)
            # Free the forward all-gather buffer. Its storage is shared with the
            # weight autograd saved for backward, so resizing to 0 releases that
            # shared allocation; the pre-backward hook re-gathers into the same
            # storage.
            self._release_full_param_storage(param_state.full_param)

        next_index = self._layer_index[layer] + self._prefetch_window_size
        if next_index < len(self._fsdp_layers):
            self._prefetch_layer_forward(self._fsdp_layers[next_index])

    def _register_forward_hooks(self) -> None:
        """
        Register the forward hooks for the FSDP layers.
        """
        for layer in self._fsdp_layers:
            self._forward_hook_handles.append(layer.register_forward_pre_hook(self._pre_forward_hook))
            self._forward_hook_handles.append(layer.register_forward_hook(self._post_forward_hook))

    def _pre_backward_hook(
        self,
        layer: torch.nn.Module,
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        """
        Pre-backward hook for the FSDP layers.
        We should gather the parameters before the backward pass.
        """
        # Self-heal the first `window` layers of the backward pass: no later
        # layer prefetched them, so gather them just-in-time here. For layers
        # already prefetched by the chain below, this is a no-op because the
        # storage is already allocated.
        self._prefetch_layer_backward(layer) # if the layer is already prefetched, this is a no-op
        self._use_prefetched_layer_backward(layer)

        prev_index = self._layer_index[layer] - self._prefetch_window_size
        if prev_index >= 0:
            self._prefetch_layer_backward(self._fsdp_layers[prev_index])

    def _register_backward_hooks(self) -> None:
        """
        Register the backward hooks for the FSDP layers.
        """
        for layer in self._fsdp_layers:
            self._backward_hook_handles.append(layer.register_full_backward_pre_hook(self._pre_backward_hook))


    def _sync_grads_of_replicated_parameters(self) -> None:
        """
        Sync the gradients of the replicated parameters.
        The replicated parameters are not sharded, so we can just all-reduce the gradients.
        """
        if self.world_size == 1:
            return
        for param in self._replicate_parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad.div_(self.world_size)
            

    def forward(self, *inputs, **kwargs):
        """
        Prefetch the first FSDP layers and cast model inputs once for mixed
        precision. Activations remain in compute_dtype unless the model itself
        explicitly changes their dtype.
        """
        for layer in self._fsdp_layers[:self._prefetch_window_size]:
            self._prefetch_layer_forward(layer)

        if self.compute_dtype is not None:
            inputs = _cast_floating(inputs, self.compute_dtype)
            kwargs = _cast_floating(kwargs, self.compute_dtype)

        outputs = self.module(*inputs, **kwargs)

        return outputs

    def finish_gradient_synchronization(self):
        """
        Wait for all the reduce scatter operations to complete.
        """
        while self._pending_reduce_scatters:
            pending = self._pending_reduce_scatters.pop(0)
            self._finalize_reduce_scatter(pending)

        self._sync_grads_of_replicated_parameters()
