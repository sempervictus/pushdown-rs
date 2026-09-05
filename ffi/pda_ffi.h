/* pda_ffi.h - the PDA device interface (the the C ABI contract).
 *
 * This is the reference FFI interface the attention-rs CUDA kernels
.
 * The dpda-rs crate produces the payload (the the pod_bytes); the attention-rs
 * consumes it (the the H2D upload + the kernels). The xinfer orchestrates.
 *
 * The layout-identity invariant: the CUDA kernels, the SIMD mock, and the
 * scalar reference all consume the SAME pod layout, so the CPU result is the
 * correctness oracle for the GPU.
 */
#ifndef PDA_FFI_H
#define PDA_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The POD header (the the first 28 bytes of the upload payload). Must match
 * the dpda-rs's PodHeader exactly (the the field order + the sizes). */
typedef struct {
    uint32_t num_states;
    uint32_t num_inputs;
    uint32_t num_stack_syms;
    uint32_t num_transitions;
    uint32_t num_accepting;
    uint32_t start_state;
    uint32_t start_stack;
} PdaPodHeader;

/* A handle to an uploaded PDA table (the the device memory). */
typedef void* PdaHandle;

/* The sampling mode (the the single-token sampling). */
typedef enum {
    PDA_SAMPLE_GREEDY = 0,
    PDA_SAMPLE_TOPK_TOPP = 1
} PdaSamplingMode;

/* The per-sequence state (the the config + the stack + the pointer). */
typedef struct {
    uint32_t ctrl;        /* the the (q_lr, q_lex) product control state */
    uint32_t stack[8];   /* the the bounded pushdown (the the D = 8) */
    uint32_t sp;         /* the the stack pointer */
} PdaSeqState;

/* Upload the PDA table to the device (the the H2D copy). Returns a handle. */
PdaHandle pda_upload(const uint8_t* pod_bytes, size_t len, int device_ordinal);

/* Free the handle (the the D2D cleanup). */
void pda_free(PdaHandle h);

/* The DP-seq mask (the the VPP NODE 1): the compute the B sequences' masks.
 * The configs_in is the B PdaSeqState; the masks_out is the B x words_per_vob
 * (the the signed VOB, the the allow/deny bits). One thread per sequence. */
void pda_compute_masks(PdaHandle h, const PdaSeqState* configs, int batch,
                       uint32_t* masks_out, int words_per_vob);

/* The Per-seq sample (the the VPP NODE 2): the advance the B sequences by the
 * sampled tokens. The tokens_in is the B uint32; the next_configs_out is the B
 * PdaSeqState. One thread per sequence. */
void pda_advance(PdaHandle h, const PdaSeqState* configs, const uint32_t* tokens,
                 int batch, PdaSeqState* next_configs_out);

/* The Per-seq projection (the the VPP NODE 3): the emit the B x (K+1) masks for
 * the K draft tokens (the the MTP/DFlash2 drafting). The drafts_in is the B x K
 * uint32; the masks_out is the B x (K+1) x words_per_vob. One thread per
 * (sequence, position).
 *
 * CRITICAL: this is a DPDA operation, NOT a DFA. Each of the K draft steps
 * advances the pushdown (the the stack[8] + the sp in the PdaSeqState): a shift
 * pushes, a reduce pops + pushes the goto. The K+1 masks are the masks at the K+1
 * successive pushdown configs. A DFA has no stack, so it cannot project forward
 * through the nesting - this is exactly why the DPDA (not the DFA) is required
 * for the drafting use-case. */
void pda_project_masks(PdaHandle h, const PdaSeqState* configs,
                      const uint32_t* drafts, int batch, int k,
                      uint32_t* masks_out, int words_per_vob);

/* The fused sampling (the the mask + the sample in one kernel, the the
 * attention-rs's optimal path). The logits_in is the B x vocab; the
 * sampled_out is the B uint32. The mode selects the greedy vs-k/top-p. */
void pda_sample_masked(PdaHandle h, const PdaSeqState* configs,
                       const float* logits, int batch, int vocab,
                       PdaSamplingMode mode, float temperature, int top_k,
                       float top_p, uint32_t* sampled_out);

#ifdef __cplusplus
}
#endif

#endif /* the PDA_FFI_H */