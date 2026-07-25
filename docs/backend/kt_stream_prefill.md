# KTransformers stream-loading prefill prototype

This branch has two different GPU-prefill implementations. They should not be
treated as equivalent:

- The existing `--kt-gpu-prefill-token-threshold` implementation came through
  kvcache-ai SGLang PR #19, its V4-Flash repair in PR #41, and the corresponding
  KTransformers submodule bump in PR #1975. `SharedFullContext` allocates an
  entire temporary expert layer, writes every expert into it, synchronizes, and
  then computes. Its expert-level host-write/H2D pipeline reduces staging
  overhead, but transfer of layer `L` does not overlap GPU compute of
  sub-layer `L`.
- `--kt-stream-prefill` selects the bounded sub-layer ring in
  `kt_stream_prefill.py`. Two persistent expert chunks have independent ready
  and consumed CUDA events. A single loader thread prepares the next chunk
  while the model stream computes the current chunk, and waits for a consumed
  event before reusing a slot.

The first implementation deliberately fails closed outside the measured
GLM-5.2 target:

- one host, pipeline parallel size 1, tensor parallel size 2;
- two explicit, distinct KT NUMA pools;
- AMXINT4 host experts, zero resident GPU experts, zero deferred experts;
- no dynamic expert update or expert LoRA;
- BF16 temporary GPU expert weights;
- exactly two ring slots and a power-of-two chunk of at most 16 experts.

For GLM-5.2 (`H=6144`, routed `I=2048`, TP=2), one BF16 expert shard is exactly
36 MiB: gate, up, and down are each 12 MiB per rank. The conservative RTX 3090
default is four experts per slot:

| Allocation per TP rank | Size |
| --- | ---: |
| One four-expert GPU slot | 144 MiB |
| Two-slot GPU ring | 288 MiB |
| Two POSIX-shared pinned host chunks | 288 MiB |
| Required additional free-VRAM margin | 512 MiB |
| Full streamed expert payload per layer | 9 GiB |

The startup shape is:

```bash
--pipeline-parallel-size 1 \
--tensor-parallel-size 2 \
--kt-method AMXINT4 \
--kt-threadpool-count 2 \
--kt-numa-nodes 0 1 \
--kt-num-gpu-experts 0 \
--kt-max-deferred-experts-per-token 0 \
--kt-gpu-prefill-token-threshold 4096 \
--kt-stream-prefill \
--kt-stream-prefill-experts-per-chunk 4 \
--kt-stream-prefill-ring-slots 2 \
--kt-stream-prefill-safety-margin-mb 512
```

Short prefill and decode remain on the ordinary KTransformers AMXINT4 path.
Only calls at or above the configured threshold enter the ring. The
authoritative quantized weights are not recreated; the new AMX export method
dequantizes one already-resident expert chunk to BF16 staging memory.

## SmallEP status

`small_ep.py` contains an executable two-rank collective contract:

1. all-gather rank-major unsorted hidden states;
2. gate independently on both ranks;
3. mask/remap routes to an explicit local expert ownership table;
4. compute a hidden-size local weighted partial;
5. all-reduce those partials and retain the original context stripe.

Its indexing, weighting, variable stripe lengths, and reduction have CPU
reference tests. It is not yet selected by the GLM model. The remaining
model-level work is to combine context-parallel attention with full (not
tensor-sliced) local expert ownership and a fused local-expert runner. Enabling
ordinary SGLang EP or TP does not implicitly provide SmallEP.

## Intra-node prefill/decode planner

`--enable-glm52-intra-node-pd-planner` installs a tokenizer-manager-owned
admission controller. It derives the long-prompt boundary from
`--kt-gpu-prefill-token-threshold` and emits one of three executable-facing
plans:

- short prompt: run chunked prefill on the unified TP=2 worker;
- isolated long prompt: run distributed SLP on both GPUs, but only with a
  successful bounded-SLP admission receipt and live capacity proof;
- long prompt while decode is active: run TP=1 SLP on one GPU, transfer KV,
  and continue TP=1 decode on the other GPU, but only with explicit TP=1
  executor, KV-transfer, shared-host-weight, and per-role capacity proofs.

The controller atomically accounts for SLP and decode reservations and exposes
external-state, prefill-complete, and release hooks. The policy code and the
existing TP=2 SLP executor are executable. The TP=1+TP=1 process/model wiring,
shared KT host-weight attachment, and KV handoff are not implemented; the
split route therefore fails closed unless a future launcher supplies all of
those contracts. The planner flag does not launch workers or alter request
routing by itself.
