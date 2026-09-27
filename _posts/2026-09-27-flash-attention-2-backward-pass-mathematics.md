---
layout: post
title: "Flash Attention 2 Backward Pass"
date: 2026-09-27 00:00:00 +0200
description: "Deriving attention gradients, implementing tiled FlashAttention-2 backward kernels in Triton with GQA and causal attention, integrating PyTorch autograd, and benchmarking against PyTorch SDPA."
excerpt: "Deriving attention gradients, implementing tiled FlashAttention-2 backward kernels in Triton with GQA and causal attention, integrating PyTorch autograd, and benchmarking against PyTorch SDPA."
categories: [gpu-programming, triton, attention]
permalink: /flash-attention-2-backward-pass-mathematics/
---

{% include mathjax.html %}

## Scope and notation

This note continues the [forward-pass derivation]({{ "/flash-attention-2-forward-pass-in-triton/" | relative_url }}). We derive the attention gradients first, then organize them into tiles so that the full score and probability matrices never need to be stored in GPU high-bandwidth memory (HBM). We then implement the backward pass in Triton, including GQA and causal attention, connect it to PyTorch autograd, and compare its performance with PyTorch SDPA backward.

As in the forward note, $B$ is the batch size, $H$ is the number of heads, $L$ is the sequence length, and $d$ is the head dimension. We start with multi-head attention (MHA), omit the independent batch and head dimensions, and work with $Q,K,V\in\mathbb{R}^{L\times d}$.

## Backpropagation through attention

The self attention operation is defined as:

$$
O = \operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V.
$$

And for convenience, we define the attention scores as:

$$
\begin{aligned}
S &= \frac{QK^T}{\sqrt{d}} \in \mathbb{R}^{L \times L}, \\
P &= \operatorname{softmax}(S) \in \mathbb{R}^{L \times L}, \\
O &= PV \in \mathbb{R}^{L \times d}.
\end{aligned}
$$

Softmax is applied independently to each row. For a loss function $\mathcal{L}$, let $dX = \partial\mathcal{L}/\partial X$ denote the gradient of the loss with respect to a tensor $X$. The backward pass receives

$$
dO = \frac{\partial \mathcal{L}}{\partial O}
$$

from the next layer and must return $dQ$, $dK$, and $dV$.

### Gradients of the weighted sum

First, we will look at backpropagation from $O$ to $P$ and $V$. We use $i,k$ for token positions and $r$ for a feature coordinate. Note that

$$
O_{ir} = \sum_{k=1}^L P_{ik} V_{kr}.
$$

So, for $V$, using the chain rule, we have:

$$
\begin{aligned}
dV_{kr} &= \sum_{i=1}^L \frac{\partial \mathcal{L}}{\partial O_{ir}} \frac{\partial O_{ir}}{\partial V_{kr}} \\
&= \sum_{i=1}^L dO_{ir} P_{ik}.
\end{aligned}
$$

In matrix form, we have:

$$
dV = P^T dO
$$

For $P$, we have:

$$
\begin{aligned}
dP_{ik} &= \sum_{r=1}^d \frac{\partial \mathcal{L}}{\partial O_{ir}} \frac{\partial O_{ir}}{\partial P_{ik}} \\
&= \sum_{r=1}^d dO_{ir} V_{kr}.
\end{aligned}
$$

In matrix form, we have:

$$
dP = dO V^T
$$

The reduction for $dV$ is over query positions, while the reduction for $dP$ is over the $d$ output features.

### Gradient of softmax

Now let's look at backpropagation through the softmax operation. For one row, temporarily dropping the query index, we have

$$
p_j = \frac{e^{s_j}}{\sum_{k=1}^L e^{s_k}}
$$

Now looking at its derivative, we have:

$$
\begin{aligned}
\frac{\partial p_x}{\partial s_y} &= \frac{\delta_{xy} e^{s_x} \sum_{k=1}^L e^{s_k} - e^{s_x} e^{s_y}}{\left(\sum_{k=1}^L e^{s_k}\right)^2} \\
&= \delta_{xy} p_x - p_x p_y
\end{aligned}
$$

Here $\delta_{xy}$ is 1 if $x=y$ and 0 otherwise. Applying the chain rule gives

$$
\begin{aligned}
ds_j &= \sum_{k=1}^L dp_k \frac{\partial p_k}{\partial s_j} \\
&= \sum_{k=1}^L dp_k (\delta_{jk} p_k - p_k p_j) \\
&= p_j dp_j - p_j \sum_{k=1}^L p_k dp_k
\end{aligned}
$$

Restoring the query index $i$, we have:

$$
dS_{ij} = P_{ij} dP_{ij} - P_{ij} \sum_{k=1}^L P_{ik} dP_{ik}
$$

### Computing the row-wise correction without storing attention

We can define a scalar per query $i$ as:

$$
\begin{aligned}
D_i &= \sum_{k=1}^L P_{ik} dP_{ik} \\
&= \sum_{k=1}^L P_{ik} \sum_{r=1}^d dO_{ir} V_{kr} \\
&= \sum_{r=1}^d dO_{ir} \sum_{k=1}^L P_{ik} V_{kr} \\
&= \sum_{r=1}^d dO_{ir} O_{ir}.
\end{aligned}
$$

So, we have:

$$
dS_{ij} = P_{ij} (dP_{ij} - D_i)
$$

The important thing to notice in this expression for $D$ is that $\sum_{k=1}^L P_{ik} dP_{ik}$ requires a reduction over an entire attention row, which we do not keep in memory during the tiled backward pass. On the other hand, $\sum_{r=1}^d dO_{ir} O_{ir}$ requires only $O$ and its gradient $dO$, which we do have available. Since $O$ was saved during forward and $dO$ is available during backward, we can compute the entire $D$ vector before processing any attention tiles:

$$
D = \operatorname{rowsum}(O\odot dO)\in\mathbb{R}^{L}.
$$

Here $\odot$ denotes elementwise multiplication, and the row sum runs over the feature dimension. This correction accounts for the fact that changing one softmax score changes all the probabilities in its row.

### Reconstructing probabilities from the forward pass

Remember that we also saved the row-wise log-sum-exp during the forward pass:

$$
L_i = \log\sum_{j=1}^L e^{S_{ij}}.
$$

We retain the forward note's notation: plain $L$ in sequence lengths and shapes is the number of tokens, while $L_i$ denotes a saved log-sum-exp value. The loss is $\mathcal{L}$. With the saved normalization, we reconstruct each probability as

$$
P_{ij} = e^{S_{ij} - L_i} = \frac{e^{S_{ij}}}{\sum_{k=1}^L e^{S_{ik}}}.
$$

Because $L_i$ already contains the normalization over all keys, even a small tile can recover the correct probabilities. We do not apply a separate softmax within each tile: that would normalize over only the keys in that tile and produce the wrong result.

### Gradients of queries and keys

Finally, since

$$
S_{ij}=\frac{1}{\sqrt d}\sum_{r=1}^d Q_{ir}K_{jr},
$$

the chain rule gives

$$
dQ_{ir}=\frac{1}{\sqrt d}\sum_{j=1}^L dS_{ij}K_{jr},
\qquad
dK_{jr}=\frac{1}{\sqrt d}\sum_{i=1}^L dS_{ij}Q_{ir}.
$$

In matrix form,

$$
\begin{aligned}
dQ &= \frac{dS K}{\sqrt{d}} \\
dK &= \frac{dS^T Q}{\sqrt{d}}
\end{aligned}
$$


## Tiling the backward pass

The equations above describe full matrices, but we can compute their contributions one tile at a time. The forward pass saves $Q,K,V,O$ and the row-wise log-sum-exp values. Before the tiled backward kernels run, a separate preprocessing step computes $D=\operatorname{rowsum}(O\odot dO)$.

### Tile notation and shapes

We use the same tile notation as the forward note. Split $Q,O,dO$ into $T_q=\lceil L/B_q\rceil$ query tiles, and $K,V$ into $T_k=\lceil L/B_k\rceil$ key-value tiles. From this point on, $i$ and $j$ index tiles rather than individual tokens. In particular, $L_i$ and $D_i$ now denote vectors containing one scalar per row of query tile $i$.

| Quantity | Shape | Meaning |
|---|---|---|
| $Q_i,O_i,dO_i$ | $B_q\times d$ | Query, output, and incoming-gradient tiles |
| $K^{(j)},V^{(j)}$ | $B_k\times d$ | Key and value tiles |
| $L_i,D_i$ | $B_q$ | Saved log-sum-exp and softmax-gradient correction |
| $S_i^j,P_i^j,dP_i^j,dS_i^j$ | $B_q\times B_k$ | Temporary score, probability, and gradient tiles |
| $dQ_i$ | $B_q\times d$ | Query-gradient tile |
| $dK^{(j)},dV^{(j)}$ | $B_k\times d$ | Key- and value-gradient tiles |

These are nominal tile sizes; a final partial tile needs padding and boundary masks in an implementation. As in the forward note, we do not tile along the feature dimension.

### Computing one tile's contributions

Given query tile $i$ and key-value tile $j$, first reconstruct the scores and probabilities:

$$
S_i^j=\frac{1}{\sqrt d}Q_i\left(K^{(j)}\right)^T,
\qquad
P_i^j=\exp\left(S_i^j-L_i\right).
$$

Subtracting $L_i$ means broadcasting its $B_q$ entries across the $B_k$ columns, or `Li[:, None]` in code. Unlike the forward pass's $\widetilde{P}_i^j$, this $P_i^j$ is already normalized over the complete key sequence.

We can then compute

$$
dP_i^j=dO_i\left(V^{(j)}\right)^T,
\qquad
dS_i^j=P_i^j\odot\left(dP_i^j-D_i\right),
$$

where $D_i$ is broadcast across columns in the same way. The tile contributes to the three gradients as follows:

$$
\begin{aligned}
dV^{(j)} &\mathrel{+}= \left(P_i^j\right)^T dO_i, \\
dK^{(j)} &\mathrel{+}= \frac{1}{\sqrt d}\left(dS_i^j\right)^T Q_i, \\
dQ_i &\mathrel{+}= \frac{1}{\sqrt d}dS_i^j K^{(j)}.
\end{aligned}
$$

Unlike the forward pass, backward does not need running maxima or online normalization updates. The saved $L_i$ supplies the complete row normalization, and the precomputed $D_i$ supplies the complete row correction for the softmax derivative. Only the gradient sums remain to be accumulated across tiles.

### Choosing which gradient tile a program owns

The three updates have different reduction directions:

$$
\begin{aligned}
dQ_i &= \frac{1}{\sqrt d}\sum_{j=1}^{T_k}dS_i^j K^{(j)}, \\
dK^{(j)} &= \frac{1}{\sqrt d}\sum_{i=1}^{T_q}\left(dS_i^j\right)^T Q_i, \\
dV^{(j)} &= \sum_{i=1}^{T_q}\left(P_i^j\right)^T dO_i.
\end{aligned}
$$

For $dQ_i$, we fix a query tile and sum over key-value tiles. For $dK^{(j)}$ and $dV^{(j)}$, we fix a key-value tile and sum over query tiles.

If each program owned a query tile and updated all three gradients, different programs would contribute to the same $dK^{(j)}$ and $dV^{(j)}$ locations. Those shared updates would require atomic additions or an additional reduction of partial results.

We instead use two passes:

| Pass | Tile owned by one program | Inner loop | Gradients written |
|---|---|---|---|
| Key/value gradients | Key-value tile $j$ | Query tiles $i=1,\ldots,T_q$ | $dK^{(j)},dV^{(j)}$ |
| Query gradients | Query tile $i$ | Key-value tiles $j=1,\ldots,T_k$ | $dQ_i$ |

Each program keeps its gradient accumulators on chip and writes them once after its inner loop. The outer loops can run in parallel across tiles, batches, and MHA heads because each program writes a distinct output region.

### Complete tiled schedule

The following is pseudocode for non-causal attention. `load`, `store`, and tile iterators stand for the corresponding memory operations and program-grid assignments; this is not an executable Triton kernel. All gradient accumulators start at zero.

```python
# Preprocessing: reduce over features, one scalar per query row.
D = rowsum(O * dO)

# Pass 1: one program per key-value tile, batch, and head.
for j in parallel_key_value_tiles:
    Kj, Vj = load(K_tile(j)), load(V_tile(j))
    dKj, dVj = zeros_like(Kj), zeros_like(Vj)

    for i in query_tiles:
        Qi, dOi = load(Q_tile(i)), load(dO_tile(i))
        Li, Di = load(L_tile(i)), load(D_tile(i))

        Sij = (Qi @ Kj.T) * scale
        Pij = exp(Sij - Li[:, None])
        dVj += Pij.T @ dOi
        dPij = dOi @ Vj.T
        dSij = Pij * (dPij - Di[:, None])
        dKj += (dSij.T @ Qi) * scale

    store(dK_tile(j), dKj)
    store(dV_tile(j), dVj)

# Pass 2: one program per query tile, batch, and head.
for i in parallel_query_tiles:
    Qi, dOi = load(Q_tile(i)), load(dO_tile(i))
    Li, Di = load(L_tile(i)), load(D_tile(i))
    dQi = zeros_like(Qi)

    for j in key_value_tiles:
        Kj, Vj = load(K_tile(j)), load(V_tile(j))

        Sij = (Qi @ Kj.T) * scale
        Pij = exp(Sij - Li[:, None])
        dPij = dOi @ Vj.T
        dSij = Pij * (dPij - Di[:, None])
        dQi += (dSij @ Kj) * scale

    store(dQ_tile(i), dQi)
```

![Flash Attention Backward Pass]({{ "/assets/triton/flash_attention_backward.png" | relative_url }})

Here `scale` is $1/\sqrt d$, and `L_tile(i)` refers to the saved log-sum-exp values. A Triton implementation should accumulate $D$ and the gradient sums in FP32, with appropriate operand casts for the matrix multiplications, as in the forward kernel.

This schedule reconstructs $P_i^j$ and $dS_i^j$ twice, once in each pass. The extra arithmetic avoids atomic gradient updates between programs. It is one way to schedule FlashAttention-2 backward; the mathematical gradients do not depend on this choice of schedule.

## Quick Notes

For GQA, a key-value head is used for multiple query heads.
Because of that, in the flash_attention_backward_dkv_kernel, we run the grid over the key-value heads, and then sum up the contributions from all the query heads.

Just like we did in the forward pass, we can use the causality mask to avoid computing parts that do not contribute to the output.
Note that we get no contribution from tiles where the $P_{ij}$ is 0 which also makes $dS_{ij}$ 0.
This happens for key index > query index.

While computing the $dQ$, the query index is fixed and we sweep over the key index, so we can use the same upper bound logic we have used in the forward pass.

For $dKV$, we have a fixed key-value index and we sweep over the query index, so in this case we need to pass the query indexes that are too small.
We also set the offsets of the blocks accordingly.

## Implementation

The implementation in [flash_attention.py](https://github.com/ozgurcelik/technical-notes/blob/main/code/flash_attention.py) follows the two-pass schedule above, with a separate kernel for the row-wise correction. The entry point is `FlashAttentionFunc`, which connects the forward and backward kernels to PyTorch autograd.

We use `N_QUERIES` and `N_KEYS` for the query and key sequence lengths, and `Hq` and `Hk` for their head counts. The tensor shapes are

$$
Q,O,dO,dQ\in\mathbb{R}^{B\times H_q\times N_q\times d},
\qquad K,V,dK,dV\in\mathbb{R}^{B\times H_k\times N_k\times d}.
$$

The saved log-sum-exp and the correction tensor both have shape $(B,H_q,N_q)$. The code supports MHA, GQA, and MQA, with $H_q$ divisible by $H_k$. The snippets below show the main operations; pointer setup and kernel arguments are omitted where they follow the same pattern as the forward note.

### Preprocessing the row-wise correction

`preprocess_kernel` computes $D_i=\sum_r O_{ir}dO_{ir}$ before either gradient pass begins. Its launch grid is

```python
Bq = 32
grid = (triton.cdiv(Nq, Bq), B * Hq)
```

Each program owns a query tile for one batch and query head:

```python
query_tile_index = tl.program_id(0)
head_q_index = tl.program_id(1) % Hq
batch_index = tl.program_id(1) // Hq
```

The block pointers for `O` and `dO` select `(Q_TILE_SIZE, d)` elements at query offset `query_tile_index * Q_TILE_SIZE`. The output pointer selects the corresponding `Q_TILE_SIZE` entries of `D`.

```python
Oi = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
D = tl.sum(Oi.to(tl.float32) * dOi.to(tl.float32), axis=-1)
tl.store(D_block_ptr, D, boundary_check=(0,))
```

Both operands are converted to FP32 before multiplication and reduction, and the correction tensor is allocated in FP32. This kernel only reads the output and its gradient, so it requires $O(BH_qN_qd)$ work and stores one scalar per query row.

### Computing key and value gradients

`flash_attention_backward_dkv_kernel` owns a key-value tile. The first grid dimension selects the tile, while the second combines batch and key-value head:

```python
key_tile_index = tl.program_id(0)
head_kv_index = tl.program_id(1) % Hk
batch_index = tl.program_id(1) // Hk
GROUP_SIZE: tl.constexpr = Hq // Hk
```

The program loads `Kj` and `Vj` once and keeps two FP32 accumulators of shape `(K_TILE_SIZE, d)` on chip. For GQA, it must accumulate contributions from every query head sharing this key-value head:

```python
dKj = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)
dVj = tl.zeros((K_TILE_SIZE, d), dtype=tl.float32)

for group_offset in range(GROUP_SIZE):
    head_q_index = head_kv_index * GROUP_SIZE + group_offset
    # Initialize Q, dO, D, and L pointers for this query head.
    # Sweep the reachable query tiles and update dKj and dVj.
```

The query-side pointers are recreated at the beginning of every head's sweep. Otherwise, after processing one head, they would still point past its final query tile. The accumulators remain live across all heads in the group; MHA is the special case `GROUP_SIZE == 1`.

For each query tile, the kernel loads `Qi`, `dOi`, `Di`, and `Li`, then performs the following updates after score masking:

```python
Sij = tl.dot(Qi, Kj.T, input_precision="ieee") * scale
Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
# Apply the causal mask here for overlapping tiles, as described below.
Pij = tl.exp(Sij - Li[:, None])
dVj = dVj + tl.dot(Pij.T.to(dOi.dtype), dOi, input_precision="ieee")
dPij = tl.dot(dOi, Vj.T, input_precision="ieee")
dSij = Pij * (dPij - Di[:, None])
dKj = dKj + tl.dot(dSij.T.to(Qi.dtype), Qi, input_precision="ieee")

Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))
L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
```

`Li` contains the natural-log log-sum-exp saved by forward, so `tl.exp(Sij - Li[:, None])` reconstructs the normalized probabilities directly. There is no running maximum or denominator in this loop.

After processing all query tiles and shared query heads, we apply the score scale to `dKj` once and store both gradients:

```python
dKj *= scale
tl.store(dK_block_ptr, dKj.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
tl.store(dV_block_ptr, dVj.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))
```

The multiplication by `scale` is needed for $dK$ because $S=\text{scale}\cdot QK^T$. It is not needed for $dV$, which comes directly from $O=PV$. Each key-value gradient tile has a single writer, even for GQA, so no atomic additions are required.

### Computing query gradients

`flash_attention_backward_dq_kernel` instead owns a query tile and loops over key-value tiles. It maps the query head to its shared key-value head using the same mapping as forward:

```python
query_tile_index = tl.program_id(0)
head_q_index = tl.program_id(1) % Hq
batch_index = tl.program_id(1) // Hq
head_kv_index = head_q_index // (Hq // Hk)
```

`Qi`, `dOi`, `Di`, and `Li` are loaded once. The inner loop streams through `Kj` and `Vj`, reconstructs the same probability and score-gradient tiles, and accumulates into `dQi`:

```python
dQi = tl.zeros((Q_TILE_SIZE, d), dtype=tl.float32)

# Inside the key-value tile loop, after loading Kj and Vj:
Sij = tl.dot(Qi, Kj.T, input_precision="ieee") * scale
k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
Sij = tl.where(k_offsets[None, :] < N_KEYS, Sij, -float('inf'))
# Apply the causal mask here for overlapping tiles.
Pij = tl.exp(Sij - Li[:, None])
dPij = tl.dot(dOi, Vj.T, input_precision="ieee")
dSij = Pij * (dPij - Di[:, None])
dQi = dQi + tl.dot(dSij.to(Kj.dtype), Kj, input_precision="ieee")

K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

# After the complete loop:
dQi *= scale
tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))
```

This pass does not sum across query heads: each query head has its own $dQ$. Recomputing `Pij` and `dSij` lets the program finish its entire gradient tile independently.

### Causal loop bounds and stages

The causal condition is `query_index >= key_index`. This implementation uses that index origin for rectangular inputs too; it does not introduce a cached-decoding position offset. The tile indices in the following code are zero-based.

For $dQ$, the query tile is fixed. As in forward, we stop after the last reachable key tile and divide the remaining tiles into fully visible and overlapping regions:

```python
Tk = tl.minimum(
    tl.cdiv(N_KEYS, K_TILE_SIZE),
    tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE),
)
split = tl.minimum(query_tile_index * Q_TILE_SIZE // K_TILE_SIZE, Tk)
```

Stage 0 visits key tiles `[0, split)` without a causal mask. Stage 1 visits `[split, Tk)` with an elementwise mask. Future tiles are skipped entirely.

For $dK,dV$, the key tile is fixed and the sweep runs over queries. We skip query tiles lying entirely before the first key in the tile:

```python
Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
q_tile_start = tl.minimum(
    Tq,
    (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE,
)
split = tl.minimum(
    tl.cdiv((key_tile_index + 1) * K_TILE_SIZE, Q_TILE_SIZE), Tq,
)
```

Here stage 0 visits overlapping query tiles `[q_tile_start, split)` with a mask. Stage 1 visits later query tiles `[split, Tq)` without one. All four query-side block pointers start at `q_tile_start * Q_TILE_SIZE` and advance continuously across both stages. If `q_tile_start == Tq`, both ranges are empty and the program stores zero gradients. A key tile beyond the last valid query can also overlap a partially padded query tile; masking and zero-padded loads still make its gradients zero.

In both passes, overlapping tiles apply

```python
q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
mask = q_offsets[:, None] >= k_offsets[None, :]
Sij = tl.where(mask, Sij, -float('inf'))
```

where `i` is the current query tile, or `query_tile_index` in the $dQ$ kernel. Setting scores to $-\infty$ makes both `Pij` and `dSij` zero at masked positions. With unequal tile sizes, the overlapping region can contain more than one tile; the floor and ceiling bounds preserve all contributions.

## Benchmark results

The following figure compares the Triton backward pass with PyTorch SDPA backward for causal MHA in FP16:

![Causal FlashAttention backward latency compared with PyTorch SDPA in FP16]({{ "/assets/triton/flash_attention_backwards_results.png" | relative_url }})

This benchmark uses batch size $B=2$, $H_q=H_k=4$, head dimension $d=64$, and sequence lengths from 128 to 8192. The vertical axis is backward latency in milliseconds, so lower bars are better.

Both providers are timed through `torch.autograd.grad` with retained forward graphs. Forward runs outside the timed region; the Triton timing includes gradient allocation and preprocessing. Compilation and autotuning are warmed up before measurement, and all three gradients are compared with PyTorch using `atol=rtol=2e-2`. PyTorch uses its default SDPA backend selection.

The plotted latencies are close at sequence lengths 256 and 512, where Triton is slightly faster, and at 1024, where PyTorch is slightly faster. At 4096, Triton takes about 1.36 ms compared with 0.97 ms for PyTorch; at 8192, the comparison is 4.73 ms versus 3.43 ms. That is roughly 1.4 times the PyTorch latency at the two largest sizes. At 128, the displayed values are 0.08 ms and 0.03 ms, so there is also a gap at the smallest size.

The implementation avoids quadratic intermediate storage and atomic gradient updates, but these results show that those properties alone do not match PyTorch's backward performance at every size. The most significant contribution to the gap is the recomputation of the probabilities and score gradients in the backward passes for query and key-value tiles.
