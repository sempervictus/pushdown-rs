# The device plan (the SIMD + the CUDA)

The device-side architecture. The SIMD interfaces + logic are complete; the CUDA
kernels consume the same bitvec layout (the layout-identity invariant: the
CUDA kernel == the SIMD batch == the scalar, the same bitvec).

## The streaming model (the batched)

The PDA step is a streaming operator: a batched pipeline node. Each node
processes the whole batch (the B items) before the next node. The batch is the
SIMD vector (the CPU) or the thread block (the GPU).

```
   input: the (config, token) pairs, the B batch
      |
      v
   +--------+    +--------+    +----------+
   | NODE 1 |    | NODE 2 |    | NODE 3  |
   | mask   |--->| step   |--->| project|
   +--------+    +--------+    +---------+
      |
      v
   output: the (next-config, mask, projected-masks) pairs, the B batch
```

- NODE 1 (the mask): the B configs -> the B masks (the O(vocab) scan, the
  SIMD-able).
- NODE 2 (the step): the B (config, token) -> the B next-configs (the
  O(1) lookup, the batched).
- NODE 3 (the project): the B configs x the K drafts -> the B x (K+1) masks
  (the drafting).

## The architecture boundary (the who-owns-what)

- pushdown-rs (this crate): produces the PdaMachine + the CudaPackage (the
  bitvec + the source primitives) + the reference FFI kernels. It has NO CUDA
  awareness beyond the bitvec layout + the FFI signatures.
- llguidance: uses pushdown-rs to generate the PDA from the CGrammar (the
  native). It has NO CUDA awareness; the runtime harnesses it to feed the table.
- attention-rs: owns the CUDA. It consumes the bitvec EITHER via the reference
  FFI kernels (the correctness oracle) OR by fusing the mask/advance/project ops
  into its existing sampling/drafting kernels (the optimal, the no separate
  launch + the no global round-trip). It also owns the on-device construction.
- xinfer: orchestrates. It hands llg's generated table to attention-rs + asks
  for masks (the single-token, the MTP linear, the DFlash2 parallel). It never
  touches CUDA.

## The fusion (the attention-rs's optimal path)

The reference FFI kernels are the SPEC + the oracle. The attention-rs fuses
them into its sampling/drafting kernels:
- The sampling kernel already has the logits in registers; the PDA mask (the
  codebook lookup + the signed-VOB apply) is fused in (the -inf for the
  illegal tokens) before the greedy/top-k/top-p sample.
- The MTP/DFlash2 drafting kernel computes the K+1 projected masks in-kernel +
  applies them to the K draft logit rows in the same pass.

The fusion consumes the SAME bitvec the FFI kernels do, so the FFI version is
the correctness oracle for the fused version.

## DP-1: the bitvec upload (the H2D copy)
- The CudaPackage (the bitvec + the source primitives) is the H2D payload.
- The bitvec is the flat POD (the no pointers, the no Rust types).
- The upload: the cudaMemcpy H2D of the bitvec + the per-seq state (the
  config, the stack, the sp).

## DP-2: the pda_compute_masks kernel (the per-seq mask)
- The one thread per seq (the B threads for the B batch).
- The input: the per-seq config (the q_lr, the q_lex, the stack) + the
  bitvec.
- The output: the per-seq mask (the VOB, the allow/deny bits).

## DP-3: the pda_advance kernel (the per-seq step)
- The one thread per seq.
- The input: the per-seq config + the sampled token.
- The output: the next per-seq config (the q_lr', the q_lex', the stack').
- The kernel: the read the config, the read the token, the lookup the
  transition (the CSR + the universal), the apply the stack op (the
  pop/push), the write the next config.

## DP-4: the pda_project_masks kernel (the K+1 masks, the drafting)
- The one thread per seq.
- The input: the per-seq config + the K draft tokens.
- The output: the K+1 per-position masks (the drafting use-case, the
  MTP/DFlash2).
- The kernel: the K sequential steps (the pushdown), the emit the K+1 masks.

## DP-5: the on-device construction (the SWYB on-the-fly config database)
- OWNED BY THE attention-rs (it owns the CUDA). The pushdown-rs
  provides the source primitives (the LR states, the lexer DFA, the
  the tokenizer bytes, the signature partition) as the upload payload; the
  attention-rs builds the per-config masks on the GPU (the batched, the
  parallel).
- The SWYB's on-the-fly config database (the configs absent from the
  precomputed database are computed on the fly + stored).

## DP-6: the FFI interface (the extern "C", the C ABI)
- The pda_upload(bitvec_ptr, len) -> the handle.
- The pda_compute_masks(handle, configs, masks_out).
- The pda_advance(handle, configs, tokens, next_configs_out).
- The pda_project_masks(handle, configs, drafts, masks_out).
- The bitvec is the C-ABI-safe payload (the flat bytes).

See ffi/pda_ffi.h for the exact C signatures.

## The CUDA graph (the graph.rs)

The PdaGraph captures the kernel sequence as a DAG (the H2D -> the
ComputeMasks -> the Sample -> the Advance -> the D2H). The CUDA graph
optimization captures this DAG once + replays it (the no per-step launch
overhead). The graph.rs defines the GraphNode enum (the H2D, the ComputeMasks,
the Sample, the Advance, the ProjectMasks, the D2H). The graph.edges are the
dependencies. The topo_order validates the DAG. The mock_replay simulates the
replay on the CPU (the layout-identity oracle).

## The three-way proof (the layout-identity)

1. The scalar (the accepts_dpda/mask_bits) == the independent oracle (the
   language correctness).
2. The SIMD batch (the step_batch) == the scalar per-item (the batched
   invariant).
3. The CUDA kernel (the pda_step_batch) == the SIMD batch (the
   layout-identity, the same bitvec).

The three-way agreement is the proof that the CUDA port is correct (the no
GPU needed for the SIMD mock).