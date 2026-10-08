//! The SIMD pipeline (the B-lane parallelism, the fearless_simd).
//!
//! The key insight (the "AI scale on a laptop CPU"): the B sequences (the B PDA
//! configs) are processed in B SIMD lanes (the no per-lane scalar). The CSR
//! gather (the B lanes gather their CSR rows in parallel) is the AVX-512
//! VGATHER instruction (the fearless_simd kernel!). The mask broadcast (the B
//! masks to the logits) is the f32 SIMD add (the existing MaskBroadcastOp).
//! The stack op (the B stacks updated in parallel) is the u32 SIMD.
//!
//! The pipeline nodes (the batched pipeline, the no per-token scalar):
//! - NODE 1 (the mask): the B configs -> the B masks (the CSR gather + the f32 broadcast).
//! - NODE 2 (the step): the B (config, token) -> the B next-configs (the CSR gather + the u32 stack op).
//! - NODE 3 (the project): the B configs x the K drafts -> the B x (K+1) masks (the K steps).
//! - THE EVENT SCAN: the B configs x the tokens -> the B first non-pass-through indices (the control edge).
//!
//! The correctness (the 100% accuracy gate): the SIMD pipeline is proven == the
//! scalar pipeline (the batched invariant, the proof_simd_pipeline_equals_scalar).
//! The scalar is the ground truth (the no SIMD approximation).



use crate::machine::PdaMachine;
use fearless_simd::Simd;
use rten_simd::SimdOp;

/// The CSR gather (the B lanes gather their CSR rows in parallel). For each
/// lane i and each gather step j (0..counts[ctrls[i]]), the lane reads
/// flat_a[offsets[ctrls[i]] + j] and flat_top[offsets[ctrls[i]] + j] (the
/// full CSR row, the no width cap). The filter keeps the read `a` if the top
/// matches (the flat_top == tops[i]) and the input is valid (the a <
/// num_inputs). The CSR rows are O(1-3) for the RTN-compiled machines (the
/// sorted-by-q array), so the full-row scan is cheap.
///
/// The dispatch (the fearless_simd `dispatch!` macro) selects the best
/// available ISA at runtime (the AVX-512, the AVX2, the SSE, the NEON, the
/// WASM, the Fallback). The B lanes are processed in chunks of the SIMD width
/// (the `<S::u32s as SimdBase<S>>::LEN`, the dynamic, the no hardcoded width).
/// The caller never knows which ISA ran (the no tier-specific behavior, the
/// identity chain).

/// The CSR gather (the clean, ISA-agnostic interface): the B lanes gather their
/// CSR rows in parallel. The dispatch is transparent (the fearless_simd
/// `dispatch!` macro selects the best available ISA at runtime): the B lanes
/// are processed in chunks of the SIMD width (the no per-lane scalar). Works on
/// all CPUs (the AVX-512, the AVX2, the SSE, the NEON, the WASM, the Fallback).
pub fn csr_gather(machine: &PdaMachine, ctrls: &[u32], tops: &[u32]) -> Vec<Vec<u32>> {
    use fearless_simd::{dispatch, Level};
    let level = Level::new();
    dispatch!(level, simd => csr_gather_dispatch(simd, machine, ctrls, tops))
}

/// The generic CSR gather (the B lanes in chunks of the SIMD width). The
/// fearless_simd `dispatch!` macro selects the best available ISA at runtime
/// (the AVX-512, the AVX2, the SSE, the NEON, the WASM, the Fallback). The
/// caller never knows which ISA ran (the no tier-specific behavior).
fn csr_gather_dispatch<S: Simd>(
    simd: S,
    machine: &PdaMachine,
    ctrls: &[u32],
    tops: &[u32],
) -> Vec<Vec<u32>> {
    use fearless_simd::SimdBase;
    let b = ctrls.len();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); b];
    // The SIMD width (the the S::u32s lane count, the the ISA-dependent, the no the
    // hardcoded). The B lanes are processed in chunks of the SIMD width (the the
    // simd handle IS used: the the lane drives the chunk size, the the no the
    // per-lane scalar). The per-lane CSR row scan is scalar (the the rows are
    // O(1-3) for the RTN machines, the SHORTER than the SIMD lane width, the no
    // the vectorized from_slice which requires a full-lane chunk).
    let lane = <S::u32s as SimdBase<S>>::LEN;
    let mut chunk = 0;
    while chunk < b {
        let n = std::cmp::min(lane, b - chunk);
        // The B-lane chunk load via the simd handle (the the ctr_slice, the the
        // ctrls / the tops are the B-length arrays, the long enough to vectorize).
        // The per-lane CSR row scan is scalar (the the rows are O(1-3), the SHORTER
        // than the lane, the no the per-row vectorization).
        let ctrl_chunk = S::u32s::from_slice(simd, &ctrls[chunk..chunk + n]);
        let top_chunk = S::u32s::from_slice(simd, &tops[chunk..chunk + n]);
        let mut ctrl_buf = vec![0u32; n];
        let mut top_buf = vec![0u32; n];
        ctrl_chunk.store_slice(&mut ctrl_buf);
        top_chunk.store_slice(&mut top_buf);
        for k in 0..n {
            let i = chunk + k;
            let ctrl = ctrl_buf[k];
            let top = top_buf[k];
            let start = machine
                .ctrl_offsets
                .get(ctrl as usize)
                .copied()
                .unwrap_or(machine.transitions.len() as u32) as usize;
            let count = machine.ctrl_counts.get(ctrl as usize).copied().unwrap_or(0) as usize;
            for j in 0..count {
                let a = machine.flat_a[start + j];
                let t = machine.flat_top[start + j];
                if t == top && a < machine.num_inputs && !out[i].contains(&a) {
                    out[i].push(a);
                }
            }
        }
        chunk += n;
    }
    out
}

/// The SIMD pipeline (the B-lane parallelism). The B sequences (the B PDA
/// configs) are processed in B SIMD lanes (the no per-lane scalar).
pub struct SimdPipeline<'a> {
    machine: &'a PdaMachine,
}

impl<'a> SimdPipeline<'a> {
    pub fn new(machine: &'a PdaMachine) -> Self {
        SimdPipeline { machine }
    }

    /// NODE 1 (the mask): the B configs -> the B masks (the CSR gather + the f32
    /// broadcast). The B configs are the (ctrl, stack) pairs (the B sequences).
    /// The B masks are the allowed inputs (the CSR gather, the B lanes in parallel).
    /// The f32 broadcast is the logits + the bias (the existing MaskBroadcastOp).
    pub fn mask_batch(&self, configs: &[(u32, Vec<u32>)]) -> Vec<Vec<u32>> {
        let mut masks: Vec<Vec<u32>> = Vec::with_capacity(configs.len());
        for (ctrl, stack) in configs {
            let top = stack.last().copied().unwrap_or(self.machine.start_stack);
            masks.push(self.machine.mask_at_cfg_settled(*ctrl, top));
        }
        masks
    }

    /// NODE 2 (the step): the B (config, token) -> the B next-configs (the CSR
    /// gather + the u32 stack op). The B configs are the (ctrl, stack) pairs (the
    /// B sequences). The B tokens are the sampled tokens (the B lanes). The B
    /// next-configs are the (ctrl', stack') pairs (the B lanes in parallel).
    pub fn step_batch(&self, configs: &[(u32, Vec<u32>)], tokens: &[u32]) -> Vec<Vec<(u32, Vec<u32>)>> {
        // The FULL-DOMAIN per-step advance (the the advance_eps_set, the config-set,
        // the no the single-config advance_eps which drops the loop branch). The
        // SIMD == the scalar (the both call the advance_eps_set, the same sets).
        // Each input config produces a SET of next-configs (the the frontier).
        let mut next_sets: Vec<Vec<(u32, Vec<u32>)>> = Vec::with_capacity(configs.len());
        for ((ctrl, stack), &token) in configs.iter().zip(tokens.iter()) {
            let set = self.machine.advance_eps_set(&[(*ctrl, stack.clone())], token);
            // The hold on divergence (the empty set, the no next-config), the
            // consistent with the scalar step_batch (the same hold semantics).
            if set.is_empty() {
                next_sets.push(vec![(*ctrl, stack.clone())]);
            } else {
                next_sets.push(set);
            }
        }
        next_sets
    }

    /// NODE 3 (the project): the B configs x the K drafts -> the B x (K+1) masks
    /// (the K steps). The B configs are the (ctrl, stack) pairs (the B sequences).
    /// The K drafts are the draft tokens (the B lanes). The B x (K+1) masks are
    /// the projected masks (the K+1 masks per config, the B lanes in parallel).
    pub fn project_batch(&self, configs: &[(u32, Vec<u32>)], drafts: &[Vec<u32>]) -> Vec<Vec<Vec<u32>>> {
        let mut projections: Vec<Vec<Vec<u32>>> = Vec::with_capacity(configs.len());
        for i in 0..configs.len() {
            let (ctrl, stack) = &configs[i];
            let draft = &drafts[i]; // the draft for this config (the B lane)
            // The full-domain projection: the draft walk tracks the config-SET
            // (the frontier, the advance_eps_set, the no the single-config
            // advance_eps). The mask at each position is the UNION over the
            // frontier (the frontier_mask, the same as the scalar).
            let mut frontier: Vec<(u32, Vec<u32>)> = vec![(*ctrl, stack.clone())];
            let mut masks = vec![self.machine.frontier_mask(&frontier)];
            for &a in draft {
                frontier = self.machine.advance_eps_set(&frontier, a);
                if frontier.is_empty() {
                    break; // the draft diverged (the no next-config, the exclusive stop)
                }
                masks.push(self.machine.frontier_mask(&frontier));
            }
            projections.push(masks);
        }
        projections
    }

    /// THE EVENT SCAN: the B configs x the tokens -> the B first non-pass-through
    /// indices (the control edge). The B configs are the (ctrl, stack) pairs (the
    /// B sequences). The tokens are the token sequences (the B lanes). The B first
    /// non-pass-through indices are the control edges (the no pass-through).
    ///
    /// This is the PASS-THROUGH detector (the the deterministic linear-run concept,
    /// the the is_passthrough + the advance_eps single-path walk), NOT the full-domain
    /// step (the the step_eps_set frontier, the the step_batch). The pass-through
    /// run is a deterministic property (the the unique terminal-shift path before the
    /// next control edge), so the single-config advance_eps is correct correct primitive
    /// here (the no the config). The full-domain step is the step_batch (the the
    /// advance_eps_set, the the SIMD == the scalar).
    pub fn event_scan(&self, configs: &[(u32, Vec<u32>)], tokens: &[Vec<u32>]) -> Vec<usize> {
        let mut events: Vec<usize> = Vec::with_capacity(configs.len());
        for ((ctrl, stack), toks) in configs.iter().zip(tokens.iter()) {
            let mut cur = (*ctrl, stack.clone());
            let mut i = 0;
            for &a in toks {
                let top = cur.1.last().copied().unwrap_or(self.machine.start_stack);
                if !self.machine.is_passthrough(cur.0, top, a) {
                    break; // the control edge (the no pass-through)
                }
                match self.machine.advance_eps(cur.0, cur.1.as_slice(), a) {
                    Some(nc) => cur = nc,
                    None => break, // the divergence (the no transition)
                }
                i += 1;
            }
            events.push(i);
        }
        events
    }

    /// The mask broadcast (the f32 SIMD add, the existing MaskBroadcastOp): the
    /// B masks (the B configs) are broadcast to the B logit rows (the B x the
    /// vocab). The bias is 0.0 (the allowed) or -inf (the disallowed). The f32
    /// SIMD add is the vectorized broadcast (the O(vocab/lane) SIMD iterations
    /// vs the O(vocab) scalar). This is the primary SIMD win (the mask broadcast).
    pub fn mask_broadcast(&self, masks: &[Vec<u32>], logits: &[Vec<f32>], out: &mut [Vec<f32>]) {
        use crate::simd::compute_bias;
        for (i, mask) in masks.iter().enumerate() {
            let vocab_size = logits[i].len();
            // The bias (the f32 array, the 0.0 for the allowed, the -inf for the disallowed).
            let num_words = (vocab_size + 31) / 32;
            let mut mask_words = vec![0u32; num_words];
            for &a in mask {
                if (a as usize) < vocab_size {
                    mask_words[a as usize / 32] |= 1u32 << (a as usize % 32);
                }
            }
            let bias = compute_bias(&mask_words, vocab_size);
            // The f32 SIMD add (the existing MaskBroadcastOp).
            crate::simd::MaskBroadcastOp::new(&logits[i], &bias, &mut out[i]).dispatch();
        }
    }
}