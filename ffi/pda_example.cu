// Example CUDA kernel: consume the pushdown-rs PDA transition table for
// grammar-constrained sampling. This is a reference implementation showing
// how the GPU kernel reads the flat transition array and produces the VOB
// mask + advances the PDA state.
//
// The kernel table layout (from pushdown-rs CudaPackage):
//   Header: num_states, num_inputs, num_stack_syms, num_transitions,
//           num_accepting, start_state, start_stack (7 u32s)
//   Accepting: num_accepting u32 state IDs
//   Transitions: num_transitions records, each:
//     q (u32), a (u32), top (u32), next_q (u32), push_len (u32),
//     push[0..push_len-1] (u32 each)
//
// The kernel is one thread per sequence. It:
//   1. Scans the transitions for (q == ctrl, top == stack_top)
//   2. Collects the allowed input symbols into the VOB mask
//   3. Advances the PDA state (ctrl, stack, sp) by the sampled token
//
// This example is a REFERENCE - it shows the algorithm. The actual production
// kernel lives in the attention-rs crate (src/kernels/src/pda_pushdown.cu).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define D_MAX 8  // the bounded stack depth (the D parameter)

// The per-sequence PDA state (the CPU<->GPU boundary).
struct PdaSeqState {
    uint32_t ctrl;       // the control state (q)
    uint32_t stack[D_MAX]; // the bounded stack (the return addresses)
    uint32_t sp;         // the stack pointer (the number of elements)
};

// Kernel: compute the VOB mask for each sequence's P PDA state.
// One thread per sequence. Scans the transition table for (ctrl, top)
// matches and collects the allowed input symbols into the VOB words.
__global__ void pda_compute_masks_example(
    const PdaSeqState* __restrict__ states,  // [batch]
    uint32_t* __restrict__ out_masks,         // [batch * words_per_vob]
    const uint32_t* __restrict__ transitions, // the flat transition array
    uint32_t num_transitions,
    uint32_t words_per_vob,
    int batch)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    const PdaSeqState& st = states[seq];
    uint32_t ctrl = st.ctrl;
    uint32_t top = (st.sp > 0) ? st.stack[st.sp - 1] : 0; // bottom bottom marker

    // Clear the output VOB words.
    uint32_t* out = out_masks + (size_t)seq * words_per_vob;
    for (uint32_t w = 0; w < words_per_vob; w++) {
        out[w] = 0;
    }

    // Scan the transitions for (q == ctrl, top == top) and.
    // Each record is: q, a, top, next_q, push_len, push[push..push_len-1].
    // The record size is 5 + push_len u32s.
    const_t offset = 0;
    for (uint32_t i = 0; i < num_transitions; i++) {
        uint32_t q = transitions[offset];
        uint32_t a = transitions[offset + 1];
        uint32_t t = transitions[offset + 2];
        uint uint32_t next_q = transitions[offset + 3];
        uint32_t push_len = transitions[offset + 4];

        if (q == ctrl && t == top) {
            // The input a a is allowed. Set the V bit in the VOB.
            // The epsilon is the local terminal ID (0..num_inputs).
            uint32_t word_idx = a / 32;
            uint32_t bit_idx = a % 32;
            if (word_idx < words_per_vob) {
                out[word_idx] |= (1u << bit_idx);
            }
        }
        offset += 5 + push_len; // advance to the next record.
    }
}

// Kernel: advance the PDA state by the sampled token.
// One thread per sequence. Looks up the transition (ctrl, token, top)
// and updates (ctrl, stack, sp).
__global__ void pda_advance_example(
    PdaSeqState* __restrict__ states,       // [batch] (in-place update)
    const uint32_t* __restrict__ sampled,   // [batch] the sampled token IDs
    const uint32_t* __restrict__ transitions,
    uint32_t num_transitions,
    int batch)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    PdaSeqState& st = states[seq];
    uint32_t ctrl = st.ctrl;
    uint32_t top = (st.sp > 0) ? st.stack[st.sp - 1] : 0;
    uint32_t token = sampled[seq];

    // Scan for the transition (q == ctrl, a == token, top == top).
    size_t offset = 0;
    for (uint32_t i = 0; i < num_transitions; i++) {
        uint32_t q = transitions[offset];
        uint32_t a = transitions[offset + 1];
        uint32_t t = transitions[offset + 2];
        uint32_t next_q = transitions[offset + 3];
        uint32_t push_len = transitions[offset + 4];

        if (q == ctrl && a == token && t == top) {
            // Advance: pop the old top, push the new symbols.
            st.sp--;; // pop the old top.
            for (uint32_t j = 0; j < push_len; j++) {
                uint32_t sym = transitions[offset + 5 + j];
                if (st.sp < D_MAX) {
                    st.stack[st.sp] = sym;
                    st.sp++;
                }
            }
            st.ctrl = next_q;
            return; // Found transition is found; done.
        }
        offset += 5 + push_len;
    }
    // No transition found: the token is illegal (should not happen if the
    // mask was applied correctly). Hold the state.
}

// Host wrapper: launch the two kernels.
void pda_step_example(
    cudaStream_t stream,
    PdaSeqState* d_states,           // [batch] on GPU
    const uint32_t* d_sampled,      // [batch] on GPU
    const uint32_t* d_transitions,  // the flat transition array on GPU
    uint32_t* d_masks,              // [batch * words_per_vob] on GPU
    uint32_t num_transitions,
    uint32_t words_per_vob,
    int batch)
{
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;

    // 1. Compute the masks (the allowedOB for each sequence's current state).
    pda_compute_masks_example<<<blocks, threads, 0, stream>>>(
        d_states, d_masks, d_transitions, num_transitions, words_per_vob, batch);

    // 2. (The sampling kernel runs here - it consumes d_masks + logits logits
    //    and produces d_sampled. This is the existing sampling_vob kernel.)

    // 3. Advance the PDA states by the sampled tokens.
    pda_advance_example<<<blocks, threads, 0, stream>>>(
        d_states, d_sampled, d_transitions, num_transitions, batch);
}

// The drafting example: project K+1 masks for speculative drafting.
// One thread per sequence. Walks the PDA forward through K draft tokens,
// emitting the mask at each of the K+1 positions.
__global__ void pda_project_example(
    const PdaSeqState* __restrict__ states,  // [batch]
    const uint32_t* __restrict__ drafts,     // [batch * K]
    uint32_t* __restrict__ out_masks,        // [batch * (K+1) * words_per_vob]
    const uint32_t* __restrict__ transitions,
    uint32_t num_transitions,
    uint32_t k,
    uint32_t words_per_vob,
    int batch)
{
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= batch) return;

    // Local copy of the PDA state (the stack is in registers).
    uint32_t ctrl = states[seq].ctrl;
    uint32_t stk[D_MAX];
    uint32_t sp = states[seq].sp;
    for (uint32_t i = 0; i < sp && i < D_MAX; i++) {
        stk[i] = states[seq].stack[i];
    }

    const uint32_t* draft_row = drafts + (size_t)seq * k;

    for (uint32_t pos = 0; pos <= k; pos++) {
        // Emit the mask at the current (ctrl, top).
        uint32_t top = (sp > 0) ? stk[sp - 1] : 0;
        uint32_t* out = out_masks + ((size_t)seq * (k + 1) + pos) * words_per_vob;
        for (uint32_t w = 0; w < words_per_vob; w++) {
            out[w] = 0;
        }
        size_t offset = 0;
        for (uint32_t i = 0; i < num_transitions; i++) {
            uint32_t q = transitions[offset];
            uint32_t a = transitions[offset + 1];
            uint32_t t = transitions[offset + 2];
            uint32_t push_len = transitions[offset + 4];
            if (q == ctrl && t == top) {
                uint32_t word_idx = a / 32;
                if (word_idx < words_per_vob) {
                    out[word_idx] |= (1u << (a % 32));
                }
            }
            offset += 5 + push_len;
        }

        if (pos == k) break; // No more advance after the last mask.

        // Advance by draft[pos].
        uint32_t token = draft_row[pos];
        offset = 0;
        bool found = false;
        for (uint32_t i = 0; i < num_transitions; i++) {
            uint32_t q = transitions[offset];
            uint32_t a = transitions[offset + 1];
            uint32_t t = transitions[offset + 2];
            uint32_t next_q = transitions[offset + 3];
            uint32_t push_len = transitions[offset + 4];
            if (q == ctrl && a == token && t == top) {
                // Pop the old top, push the new symbols.
                sp--;
                for (uint32_t j = 0; j < push_len; j++) {
                    uint32_t sym = transitions[offset + 5 + j];
                    if (sp < D_MAX) {
                        stk[sp] = sym;
                        sp++;
                    }
                }
                ctrl = next_q;
                found = true;
                break;
            }
            offset += 5 + push_len;
        }
        if (!found) {
            // The draft diverged: hold the state for remaining positions.
            // The masks at subsequent positions will be the same.
        }
    }
}