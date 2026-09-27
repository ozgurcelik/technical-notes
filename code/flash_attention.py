# %%
# pyright: reportUnreachable=false
import triton
import triton.language as tl
import torch
from torch import Tensor
import math
from einops import einsum
from jaxtyping import Bool, Float
from typing import Tuple
# %%
def naive_attention(Q: Float[Tensor, " ... L d_h"],
                    K: Float[Tensor, " ... L d_k"],
                    V: Float[Tensor, " ... L d_k"],
                    mask: Bool[Tensor, " ... L L"] | None = None) -> Float[Tensor, " ... L d_k"]:
    """
    Naive attention implementation.
    """
    d_k = K.shape[-1]
    scores = einsum(Q, K, "... query d, ... key d -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        scores = torch.where(mask, scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return einsum(weights, V, "... query key, ... key d -> ... query d")

def pytorch_attention(Q: Float[Tensor, " ... L d_h"],
                      K: Float[Tensor, " ... L d_k"],
                      V: Float[Tensor, " ... L d_k"],
                      mask: Bool[Tensor, " ... L L"] | None = None) -> Float[Tensor, " ... L d_k"]:
    """
    PyTorch attention implementation.
    """
    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

# %%
@triton.jit
def flash_attention_forward_kernel(
    Q_ptr, #[B, Hq, Lq, D]
    K_ptr, #[B, Hk, Lk, D]
    V_ptr, #[B, Hk, Lk, D]
    O_ptr, #[B, Hq, Lq, D]
    L_ptr, #[B, Hq, Lq]
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lq: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    Hq: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1) % Hq
    batch_index = tl.program_id(1) // Hq

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # we will loop over the entire key matrix
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.zeros((Q_TILE_SIZE, triton.next_power_of_2(D)), dtype=tl.float32) # (Q_TILE_SIZE, D)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (Q_TILE_SIZE,)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32) # (Q_TILE_SIZE,)

    # Baseline: visit every key tile, including masked future tiles.
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Sij = tl.dot(Qi, Kj.T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        # Due to padding, some parts of the Sij matrix will be 0
        # But, this would mess up the softmax operation,
        # so we need to identify the padded elements and set the corresponding elements of Sij to -inf
        # now, the padding from the 0th dimension of Q is irrelevant since softmax is apllied along for each row separately, so any extra rows in S will be ignored down the line anyways
        # the padding along the 1st dimension of Q and K is also irrelevant since we will be multiplying all the elements along the D dimension of Q and K, so padded parts will be adding 0 to summation
        # But we need to take care of the padding along the 0th dimension of K, since padded parts there will be adding columns filled with 0s to the Sij matrix, which then messes up the denominator of softmax
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        if is_causal:
            q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = tl.where(mask, Sij, -float('inf'))
        mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
        Pij = tl.exp(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        alpha = tl.exp(mi - mi_new)
        li = alpha * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
        Oi = tl.dot(Pij.to(Vj.dtype), Vj, Oi * alpha[:, None]) # (Q_TILE_SIZE, D)
        mi = mi_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi * (1.0 / li[:, None])
    # Natural-log logsumexp from the running softmax state.
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

# %%
@triton.jit
def flash_attention_forward_kernel_tk_trick(
    Q_ptr, #[B, Hq, Lq, D]
    K_ptr, #[B, Hk, Lk, D]
    V_ptr, #[B, Hk, Lk, D]
    O_ptr, #[B, Hq, Lq, D]
    L_ptr, #[B, Hq, Lq]
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lq: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    Hq: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1) % Hq
    batch_index = tl.program_id(1) // Hq

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # we will loop over the entire key matrix
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.zeros((Q_TILE_SIZE, triton.next_power_of_2(D)), dtype=tl.float32) # (Q_TILE_SIZE, D)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (Q_TILE_SIZE,)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32) # (Q_TILE_SIZE,)

    # For causal attention, key tiles whose smallest key index is already past
    # the largest query index in this query tile would have Sij entirely masked
    # to -inf -- which is a no-op for the running (mi, li, Oi) state but still
    # costs two tl.loads, a tl.dot, and the mask/exp work. Tightening the loop
    # bound to skip those tiles cuts work roughly in half for causal and is
    # what makes causal attention actually faster than full attention.
    if is_causal:
        # Last reachable key index for this query tile is
        #   (query_tile_index + 1) * Q_TILE_SIZE - 1,
        # so the number of key tiles we need to visit is
        #   ceil(((query_tile_index + 1) * Q_TILE_SIZE) / K_TILE_SIZE).
        # Also clamp to the actual number of key tiles so we don't run past N_KEYS.
        Tk = tl.minimum(
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE),
            tl.cdiv(N_KEYS, K_TILE_SIZE),
        )
    else:
        Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        Sij = tl.dot(Qi, Kj.T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)
        Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
        if is_causal:
            # Earlier key tiles are fully visible; only overlapping tiles need a mask.
            if (j + 1) * K_TILE_SIZE > query_tile_index * Q_TILE_SIZE:
                q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                mask = q_offsets[:, None] >= k_offsets[None, :]
                Sij = tl.where(mask, Sij, -float('inf'))
        mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
        Pij = tl.exp(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        alpha = tl.exp(mi - mi_new)
        li = alpha * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
        Oi = tl.dot(Pij.to(Vj.dtype), Vj, Oi * alpha[:, None]) # (Q_TILE_SIZE, D)
        mi = mi_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi * (1.0 / li[:, None])
    # Natural-log logsumexp from the running softmax state.
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


@triton.jit
def flash_attention_forward_stages(
    Q_ptr, #[B, Hq, Lq, D]
    K_ptr, #[B, Hk, Lk, D]
    V_ptr, #[B, Hk, Lk, D]
    O_ptr, #[B, Hq, Lq, D]
    L_ptr, #[B, Hq, Lq]
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lq: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    Hq: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_index = tl.program_id(1) % Hq
    batch_index = tl.program_id(1) // Hq

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # we will loop over the entire key matrix
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.zeros((Q_TILE_SIZE, triton.next_power_of_2(D)), dtype=tl.float32) # (Q_TILE_SIZE, D)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (Q_TILE_SIZE,)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32) # (Q_TILE_SIZE,)

    # If the attention is not causal, then there is no need for masking anyways
    # But if it is causal, then we can have 2 stages
    # first stage: all the key tile is visible to the query tile
    # second stage: some parts of the key tile is not visible to the query tile so we need to do masking


    # Stop before fully masked future key tiles in causal attention.
    if is_causal:
        Tk = tl.minimum(
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE),
            tl.cdiv(N_KEYS, K_TILE_SIZE),
        )
    else:
        Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for stage in tl.static_range(2 if is_causal else 1):
        if is_causal:
            split = tl.minimum(query_tile_index * Q_TILE_SIZE // K_TILE_SIZE, Tk)
            lo = 0 if stage == 0 else split
            hi = split if stage == 0 else Tk
        else:
            lo = 0
            hi = Tk

        for j in range(lo, hi):
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
            Sij = tl.dot(Qi, Kj.T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)
            Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
            if is_causal and stage == 1:
                q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                mask = q_offsets[:, None] >= k_offsets[None, :]
                Sij = tl.where(mask, Sij, -float('inf'))
            mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
            Pij = tl.exp(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
            alpha = tl.exp(mi - mi_new)
            li = alpha * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
            Oi = tl.dot(Pij.to(Vj.dtype), Vj, Oi * alpha[:, None]) # (Q_TILE_SIZE, D)
            mi = mi_new
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi * (1.0 / li[:, None])
    # Natural-log logsumexp from the running softmax state.
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

@triton.jit
def flash_attention_forward_gqa_kernel(
    Q_ptr, #[B, Hq, Lq, D]
    K_ptr, #[B, Hk, Lk, D]
    V_ptr, #[B, Hk, Lk, D]
    O_ptr, #[B, Hq, Lq, D]
    L_ptr, #[B, Hq, Lq]
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lq: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_q_index = tl.program_id(1) % Hq
    batch_index = tl.program_id(1) // Hq

    head_kv_index = head_q_index // (Hq // Hk)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_q_index * stride_qh,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_kv_index * stride_kh,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0), # we will loop over the entire key matrix
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_kv_index * stride_vh,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_q_index * stride_oh,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, triton.next_power_of_2(D)),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_q_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)
    Oi = tl.zeros((Q_TILE_SIZE, triton.next_power_of_2(D)), dtype=tl.float32) # (Q_TILE_SIZE, D)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # (Q_TILE_SIZE,)
    mi = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32) # (Q_TILE_SIZE,)

    # If the attention is not causal, then there is no need for masking anyways
    # But if it is causal, then we can have 2 stages
    # first stage: all the key tile is visible to the query tile
    # second stage: some parts of the key tile is not visible to the query tile so we need to do masking


    # Stop before fully masked future key tiles in causal attention.
    if is_causal:
        Tk = tl.minimum(
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE),
            tl.cdiv(N_KEYS, K_TILE_SIZE),
        )
    else:
        Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for stage in tl.static_range(2 if is_causal else 1):
        if is_causal:
            split = tl.minimum(query_tile_index * Q_TILE_SIZE // K_TILE_SIZE, Tk)
            lo = 0 if stage == 0 else split
            hi = split if stage == 0 else Tk
        else:
            lo = 0
            hi = Tk

        for j in range(lo, hi):
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
            Sij = tl.dot(Qi, Kj.T) * scale # (Q_TILE_SIZE, K_TILE_SIZE)
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)
            Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
            if is_causal and stage == 1:
                q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                mask = q_offsets[:, None] >= k_offsets[None, :]
                Sij = tl.where(mask, Sij, -float('inf'))
            mi_new = tl.maximum(mi, tl.max(Sij, axis=-1)) # (Q_TILE_SIZE,)
            Pij = tl.exp(Sij - mi_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
            alpha = tl.exp(mi - mi_new)
            li = alpha * li + tl.sum(Pij, axis=-1) # (Q_TILE_SIZE,)
            Oi = tl.dot(Pij.to(Vj.dtype), Vj, Oi * alpha[:, None]) # (Q_TILE_SIZE, D)
            mi = mi_new
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi * (1.0 / li[:, None])
    # Natural-log logsumexp from the running softmax state.
    Li = mi + tl.log(li)
    tl.store(O_block_ptr, Oi.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

_FLASH_ATTENTION_CONFIGS = [
    triton.Config(
        {"Q_TILE_SIZE": q_tile, "K_TILE_SIZE": k_tile},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for q_tile, k_tile in ((32, 32), (64, 32), (64, 64), (128, 32), (128, 64), (128, 128))
    for num_warps in (4, 8)
    for num_stages in (2, 3)
]
flash_attention_forward_kernel_autotuned = triton.autotune(
    configs=_FLASH_ATTENTION_CONFIGS,
    key=["N_QUERIES", "N_KEYS", "D", "is_causal"],
)(flash_attention_forward_kernel)

flash_attention_forward_kernel_tk_trick_autotuned = triton.autotune(
    configs=_FLASH_ATTENTION_CONFIGS,
    key=["N_QUERIES", "N_KEYS", "D", "is_causal"],
)(flash_attention_forward_kernel_tk_trick)


flash_attention_forward_stages_autotuned = triton.autotune(
    configs=_FLASH_ATTENTION_CONFIGS,
    key=["N_QUERIES", "N_KEYS", "D", "is_causal"],
)(flash_attention_forward_stages)


flash_attention_forward_gqa_autotuned = triton.autotune(
    configs=_FLASH_ATTENTION_CONFIGS,
    key=["N_QUERIES", "N_KEYS", "D", "is_causal", "Hq", "Hk"],
)(flash_attention_forward_gqa_kernel)


def flash_attention_forward(Q: Float[Tensor, " ... L d_h"],
                            K: Float[Tensor, " ... L d_k"],
                            V: Float[Tensor, " ... L d_k"],
                            is_causal: bool = False,
                            TK_trick: bool = True,
                            autotune: bool = False,
                            stages: bool = False) -> Tuple[Float[Tensor, " ... L d_k"], Float[Tensor, " ... L"]]:
    """
    Flash attention forward pass. stages selects the staged kernel; otherwise
    TK_trick selects between the TK and baseline kernels.
    """
    B, Hq, Lq, D = Q.shape
    B, Hk, Lk, D = K.shape
    B, Hk, Lk, D = V.shape
    if Hq != Hk:
        raise ValueError(
            f"Expected MHA shapes with Hq == Hk, got Hq={Hq}, Hk={Hk}. "
            "Use flash_attention_forward_gqa for GQA/MQA."
        )
    scale = 1.0 / math.sqrt(D)
    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32
    O = torch.empty((B, Hq, Lq, D), device=Q.device, dtype=Q.dtype)
    L = torch.empty((B, Hq, Lq), device=Q.device, dtype=torch.float32)
    if stages:
        kernel = (flash_attention_forward_stages_autotuned if autotune
                  else flash_attention_forward_stages)
    elif TK_trick:
        kernel = (flash_attention_forward_kernel_tk_trick_autotuned if autotune
                  else flash_attention_forward_kernel_tk_trick)
    else:
        kernel = (flash_attention_forward_kernel_autotuned if autotune
                  else flash_attention_forward_kernel)
    grid = lambda meta: (triton.cdiv(Lq, meta["Q_TILE_SIZE"]), B * Hq)
    tile_args = {} if autotune else {
        "Q_TILE_SIZE": Q_TILE_SIZE,
        "K_TILE_SIZE": K_TILE_SIZE,
        "num_warps": 4,
        "num_stages": 3,
    }
    kernel[grid](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        Lq, Lk,
        scale,
        D,
        is_causal=is_causal, Hq=Hq, **tile_args,
    )
    return O, L


def flash_attention_forward_gqa(Q: Float[Tensor, " ... L d_h"],
                                K: Float[Tensor, " ... L d_k"],
                                V: Float[Tensor, " ... L d_k"],
                                is_causal: bool = False,
                                autotune: bool = False) -> Tuple[Float[Tensor, " ... L d_k"], Float[Tensor, " ... L"]]:
    """
    Flash attention forward pass with native GQA/MQA head mapping.
    Each query head q maps to KV head q // (Hq // Hk).
    """
    B, Hq, Lq, D = Q.shape
    _, Hk, Lk, Dk = K.shape
    _, Hv, Lv, Dv = V.shape
    if Hk != Hv or Lk != Lv or Dk != Dv or Dk != D:
        raise ValueError("K and V must match in heads, sequence length, and head dim, and match Q's head dim.")
    if Hq % Hk != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hk ({Hk}) for GQA.")
    scale = 1.0 / math.sqrt(D)
    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32
    O = torch.empty((B, Hq, Lq, D), device=Q.device, dtype=Q.dtype)
    L = torch.empty((B, Hq, Lq), device=Q.device, dtype=torch.float32)
    kernel = (flash_attention_forward_gqa_autotuned if autotune
              else flash_attention_forward_gqa_kernel)
    grid = lambda meta: (triton.cdiv(Lq, meta["Q_TILE_SIZE"]), B * Hq)
    tile_args = {} if autotune else {
        "Q_TILE_SIZE": Q_TILE_SIZE,
        "K_TILE_SIZE": K_TILE_SIZE,
        "num_warps": 4,
        "num_stages": 3,
    }
    kernel[grid](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        Lq, Lk,
        scale,
        D,
        is_causal=is_causal, Hq=Hq, Hk=Hk, **tile_args,
    )
    return O, L

# %%
@triton.jit
def preprocess_kernel(
    O_ptr, dO_ptr, #[B, Hq, Lq, d]
    D_ptr, #[B, Hq, Lq]
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
    stride_dOb: tl.constexpr, stride_dOh: tl.constexpr, stride_dOq: tl.constexpr, stride_dOd: tl.constexpr,
    stride_db: tl.constexpr, stride_dh: tl.constexpr, stride_dq: tl.constexpr,
    N_QUERIES: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    Hq: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_q_index = tl.program_id(1) % Hq
    batch_index = tl.program_id(1) // Hq

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob + head_q_index * stride_oh,
        shape=(N_QUERIES, d),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dOb + head_q_index * stride_dOh,
        shape=(N_QUERIES, d),
        strides=(stride_dOq, stride_dOd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db + head_q_index * stride_dh,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Oi = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
    D = tl.sum(Oi.to(tl.float32) * dOi.to(tl.float32), axis=-1) # (Q_TILE_SIZE,)
    tl.store(D_block_ptr, D, boundary_check=(0,))

@triton.jit
def flash_attention_backward_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, #[B, Hq, Lq, d], [B, Hk, Lk, d], [B, Hk, Lk, d]
    dO_ptr, D_ptr, L_ptr, #[B, Hq, Lq, d], [B, Hq, Lq], [B, Hq, Lq]
    dK_ptr, dV_ptr, #[B, Hk, Lk, d], [B, Hk, Lk, d]
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
    stride_dOb: tl.constexpr, stride_dOh: tl.constexpr, stride_dOq: tl.constexpr, stride_dOd: tl.constexpr,
    stride_db: tl.constexpr, stride_dh: tl.constexpr, stride_dq: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lq: tl.constexpr,
    stride_dKb: tl.constexpr, stride_dKh: tl.constexpr, stride_dKq: tl.constexpr, stride_dKd: tl.constexpr,
    stride_dVb: tl.constexpr, stride_dVh: tl.constexpr, stride_dVq: tl.constexpr, stride_dVd: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    # One program owns a key tile for one KV head; no shared output writes.
    head_kv_index = tl.program_id(1) % Hk
    batch_index = tl.program_id(1) // Hk
    tl.static_assert(Hk > 0 and Hq % Hk == 0, "Hq must be divisible by Hk")
    GROUP_SIZE: tl.constexpr = Hq // Hk

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_kv_index * stride_kh,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_kv_index * stride_vh,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dKb + head_kv_index * stride_dKh,
        shape=(N_KEYS, d),
        strides=(stride_dKq, stride_dKd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dVb + head_kv_index * stride_dVh,
        shape=(N_KEYS, d),
        strides=(stride_dVq, stride_dVd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)

    k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) # (K_TILE_SIZE,)

    dKj = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
    dVj = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)

    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    q_tile_start = 0
    if is_causal:
        # Keep the query tile containing the first key, even when it overlaps.
        q_tile_start = tl.minimum(
            Tq,
            (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE,
        )
    for group_offset in range(GROUP_SIZE):
        head_q_index = head_kv_index * GROUP_SIZE + group_offset
        # Reset every query-side pointer to the first reachable tile for each head.
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb + head_q_index * stride_qh,
            shape=(N_QUERIES, d),
            strides=(stride_qq, stride_qd),
            offsets=(q_tile_start * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, d),
            order=(1, 0),
        )

        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dOb + head_q_index * stride_dOh,
            shape=(N_QUERIES, d),
            strides=(stride_dOq, stride_dOd),
            offsets=(q_tile_start * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, d),
            order=(1, 0),
        )

        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db + head_q_index * stride_dh,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(q_tile_start * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb + head_q_index * stride_lh,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(q_tile_start * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Overlapping query tiles first, then fully visible query tiles.
        # Pointers advance continuously across both stages for this query head.
        for stage in tl.static_range(2 if is_causal else 1):
            if is_causal:
                split = tl.minimum(
                    tl.cdiv((key_tile_index + 1) * K_TILE_SIZE, Q_TILE_SIZE), Tq,
                )
                lo = q_tile_start if stage == 0 else split
                hi = split if stage == 0 else Tq
            else:
                lo = 0
                hi = Tq
            for i in range(lo, hi):
                Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
                dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
                Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
                Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

                Sij = tl.dot(Qi, Kj.T, input_precision="ieee") * scale # (Q_TILE_SIZE, K_TILE_SIZE)
                Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
                if is_causal and stage == 0:
                    q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                    mask = q_offsets[:, None] >= k_offsets[None, :]
                    Sij = tl.where(mask, Sij, -float('inf'))
                Pij = tl.exp(Sij - Li[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
                dVj = dVj + tl.dot(Pij.T.to(dOi.dtype), dOi, input_precision="ieee") # (K_TILE_SIZE, d)
                dPij = tl.dot(dOi, Vj.T, input_precision="ieee") # (Q_TILE_SIZE, K_TILE_SIZE)
                dSij = Pij * (dPij - Di[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
                dKj = dKj + tl.dot(dSij.T.to(Qi.dtype), Qi, input_precision="ieee") # (K_TILE_SIZE, d)

                Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
                dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
                D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
                L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))

    # Apply the gradient scale once, after all query tiles and shared heads.
    dKj *= scale

    tl.store(dK_block_ptr, dKj.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

@triton.jit
def flash_attention_backward_dq_kernel(
    Q_ptr, K_ptr, V_ptr, #[B, Hq, Lq, d], [B, Hk, Lk, d], [B, Hk, Lk, d]
    dO_ptr, D_ptr, L_ptr, #[B, Hq, Lq, d], [B, Hq, Lq], [B, Hq, Lq]
    dQ_ptr, #[B, Hq, Lq, d]
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
    stride_dOb: tl.constexpr, stride_dOh: tl.constexpr, stride_dOq: tl.constexpr, stride_dOd: tl.constexpr,
    stride_db: tl.constexpr, stride_dh: tl.constexpr, stride_dq: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lq: tl.constexpr,
    stride_dQb: tl.constexpr, stride_dQh: tl.constexpr, stride_dQq: tl.constexpr, stride_dQd: tl.constexpr,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
    scale: tl.constexpr,
    d: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    head_q_index = tl.program_id(1) % Hq
    batch_index = tl.program_id(1) // Hq

    head_kv_index = head_q_index // (Hq // Hk)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb + head_q_index * stride_qh,
        shape=(N_QUERIES, d),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dOb + head_q_index * stride_dOh,
        shape=(N_QUERIES, d),
        strides=(stride_dOq, stride_dOd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb + head_kv_index * stride_kh,
        shape=(N_KEYS, d),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb + head_kv_index * stride_vh,
        shape=(N_KEYS, d),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, d),
        order=(1, 0),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db + head_q_index * stride_dh,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb + head_q_index * stride_lh,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dQb + head_q_index * stride_dQh,
        shape=(N_QUERIES, d),
        strides=(stride_dQq, stride_dQd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, d),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, d)
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

    dQi = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_causal:
        Tk = tl.minimum(
            Tk,
            tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE),
        )
    # Fully visible key tiles first, then tiles overlapping the causal diagonal.
    for stage in tl.static_range(2 if is_causal else 1):
        if is_causal:
            split = tl.minimum(query_tile_index * Q_TILE_SIZE // K_TILE_SIZE, Tk)
            lo = 0 if stage == 0 else split
            hi = split if stage == 0 else Tk
        else:
            lo = 0
            hi = Tk
        for j in range(lo, hi):
            Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)
            Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, d)

            Sij = tl.dot(Qi, Kj.T, input_precision="ieee") * scale # (Q_TILE_SIZE, K_TILE_SIZE)
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
            if is_causal and stage == 1:
                q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                mask = q_offsets[:, None] >= k_offsets[None, :]
                Sij = tl.where(mask, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
            dPij = tl.dot(dOi, Vj.T, input_precision="ieee") # (Q_TILE_SIZE, K_TILE_SIZE)
            dSij = Pij * (dPij - Di[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
            dQi = dQi + tl.dot(dSij.to(Kj.dtype), Kj, input_precision="ieee") # (Q_TILE_SIZE, d)

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    dQi *= scale

    tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

# Tune the two backward passes independently; their reduction axes differ.
# Causal stages above partition the diagonal; num_stages below controls pipelining.
_FLASH_ATTENTION_BACKWARD_CONFIGS = [
    triton.Config(
        {"Q_TILE_SIZE": q_tile, "K_TILE_SIZE": k_tile},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for q_tile, k_tile in (
        (32, 32), (32, 64), (64, 32), (64, 64),
        (32, 128), (128, 32),
    )
    for num_warps in (4, 8)
    for num_stages in (2, 3)
]
# Include batch/layout information as well as shape and head sharing in the key.
# Triton also includes tensor dtypes in its tuning cache key.
_FLASH_ATTENTION_BACKWARD_KEY = [
    "N_QUERIES", "N_KEYS", "d", "is_causal", "Hq", "Hk", "BATCH_SIZE",
    "stride_qb", "stride_qh", "stride_qq", "stride_qd",
    "stride_kb", "stride_kh", "stride_kk", "stride_kd",
    "stride_vb", "stride_vh", "stride_vk", "stride_vd",
]
# Each launch overwrites its outputs, so tuning needs no reset_to_zero.
flash_attention_backward_dkv_kernel_autotuned = triton.autotune(
    configs=_FLASH_ATTENTION_BACKWARD_CONFIGS,
    key=_FLASH_ATTENTION_BACKWARD_KEY,
)(flash_attention_backward_dkv_kernel)

flash_attention_backward_dq_kernel_autotuned = triton.autotune(
    configs=_FLASH_ATTENTION_BACKWARD_CONFIGS,
    key=_FLASH_ATTENTION_BACKWARD_KEY,
)(flash_attention_backward_dq_kernel)


class FlashAttentionFunc(torch.autograd.Function):
    """First-order autograd for MHA/GQA/MQA with [B, H, N, D] inputs.

    Supports FP32/FP16/BF16 and power-of-two head dimensions >= 16.
    Causal masking uses query_index >= key_index, including rectangular inputs.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        if any(t.ndim != 4 for t in (Q, K, V)):
            raise ValueError("Q, K, V must have shape [B, H, N, D]")
        B, Hq, Nq, d = Q.shape
        if (K.shape != V.shape or K.shape[0] != B or K.shape[-1] != d
                or min(B, Hq, Nq, K.shape[1], K.shape[2]) <= 0
                or Hq % K.shape[1] != 0):
            raise ValueError("Q/K/V shapes must match in batch/head dimension; Hq must be divisible by Hk")
        if d < 16 or d & (d - 1):
            raise ValueError("Backward requires a power-of-two head dimension >= 16")
        if (Q.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or any(t.device != Q.device or t.dtype != Q.dtype for t in (K, V))
                or Q.device.type != "cuda"):
            raise ValueError("Q/K/V must share a GPU device and FP32/FP16/BF16 dtype")
        forward = flash_attention_forward if Hq == K.shape[1] else flash_attention_forward_gqa
        O, L = forward(Q, K, V, is_causal=is_causal)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        B, Hq, Nq, d = Q.shape
        Hk, Nk = K.shape[1:3]
        Bq = 32  # Preprocessing has its own fixed tile size.
        # Materialize expanded/strided upstream gradients for block-pointer loads.
        dO = dO.contiguous()
        Di = torch.empty((B, Hq, Nq), device=Q.device, dtype=torch.float32)
        preprocess_kernel[(triton.cdiv(Nq, Bq), B * Hq)](
            O, dO, Di, *O.stride(), *dO.stride(), *Di.stride(), Nq, d, Bq, Hq,
        )
        strides = (*Q.stride(), *K.stride(), *V.stride(), *dO.stride(),
                   *Di.stride(), *L.stride())
        args = dict(N_QUERIES=Nq, N_KEYS=Nk, scale=d**-0.5, d=d,
                    is_causal=ctx.is_causal, Hq=Hq, Hk=Hk, BATCH_SIZE=B)
        dQ = dK = dV = None
        if ctx.needs_input_grad[0]:
            dQ = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)
            flash_attention_backward_dq_kernel_autotuned[
                lambda meta: (triton.cdiv(Nq, meta["Q_TILE_SIZE"]), B * Hq)
            ](
                Q, K, V, dO, Di, L, dQ, *strides, *dQ.stride(), **args,
            )
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dK = torch.empty(K.shape, device=K.device, dtype=K.dtype)
            dV = torch.empty(V.shape, device=V.device, dtype=V.dtype)
            flash_attention_backward_dkv_kernel_autotuned[
                lambda meta: (triton.cdiv(Nk, meta["K_TILE_SIZE"]), B * Hk)
            ](
                Q, K, V, dO, Di, L, dK, dV, *strides,
                *dK.stride(), *dV.stride(), **args,
            )
        gradients = (dQ, dK if ctx.needs_input_grad[1] else None,
                     dV if ctx.needs_input_grad[2] else None, None)
        # apply(Q, K, V) may omit the optional causal argument.
        return gradients[:len(ctx.needs_input_grad)]
