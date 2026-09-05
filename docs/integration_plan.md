# The Integration Plan (the llg -> pushdown-rs -> attention-rs -> xinfer)

The end for the PDA-based constrained decoding + drafting.

## The data flow
```
   llguidance (the no CUDA awareness)
     |  the CGrammar (the compiled grammar)
     |  the native matcher (the oracle)
     v
   pushdown-rs (the PDA generation, the CPU-side)
     |  compile(&CGrammar) -> PdaMachine (the RTN, the LR(1) + the stack)
     |  PdaMachine.to_bitvec() -> the CudaPackage (the POD table)
     |  the source primitives (the LR states, the spanner, the codebook)
     v
   attention-rs (the CUDA-native, the owns the GPU)
     |  (a) the reference FFI kernels (the pda_compute_masks/advance/project)
     |  (b) FUSED into the sampling/drafting kernels (the optimal)
     |  (c) the on-device construction (the SWYB on-the-fly, the optional)
     v
   xinfer (the runtime, the orchestrator)
     |  the single-token sampling (the greedy/top-k/top-p, the masked)
     |  the MTP linear drafting (the K+1 projected masks)
     |  the DFlash2 parallel drafting (the K+1 projected masks, the block)
```

## The interfaces (the contracts)

### llg -> pushdown-rs
- The `CGrammar` implements the `pushdown_rs::Grammar` trait (the adapter, the
  dpguidance's dpda_adapter.rs).
- The `pushdown_rs::compile(&CGrammar) -> PdaMachine` (the RTN compilation).
- The `PdaMachine.to_bitvec() -> BitVec` (the POD table).
- The `CudaPackage` (the bitvec + the source primitives, the upload payload).

### pushdown-rs -> attention-rs
- The `CudaPackage` bitvec (the flat POD, the H2D payload).
- The FFI header (the pda_ffi.h, the extern "C" kernel signatures).
- The source primitives (the LR states, the spanner, the codebook)
  for the on-device construction (the DP-5).

### attention-rs -> xinfer
- The `pda_compute_masks` (the per-seq mask, the single-token sampling).
- The `pda_sample` (the greedy/top-k/top-p with the mask).
- The `pda_project_masks` (the K+1 masks, the MTP/DFlash2 drafting).
- The `pda_advance` (the per-seq step, the state update).

## The correctness contract (the three-way)
1. The scalar (the pushdown-rs's accepts/mask_bits) == the independent oracle
   (the language correctness, the llg native matcher).
2. The SIMD batch (the step_batch/project_batch) == the scalar per-item
   (the batch invariant).
3. The CUDA kernel (the attention-rs) == the SIMD batch (the
   layout-identity, the same bitvec).

The GPU server validates (3) by running the FFI kernels + comparing to the
CPU/SIMD reference (the layout-identity). The fusion (the (-rs's
optimal path) is validated against the FFI kernels (the same bitvec, the
same algorithm).

## The build order (the what to build first)
1. The pushdown-rs: the CudaPackage + the FFI header (the this crate, the
   compiles here).
2. The attention-rs: the reference FFI kernels (the pda.cu, the needs a
   GPU to test).
3. The attention-rs: the fusion into the sampling/drafting kernels (the
   optimal, the needs a GPU to test).
4. The xinfer: the orchestration (the llg generates, the attention-rs
   executes, the xinfer drafts).