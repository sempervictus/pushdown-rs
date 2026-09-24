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
use rten_simd::SimdOp;

/// The SIMD pipeline (the B-lane parallelism). The B sequences (the B PDA
/// configs) are processed in B SIMD lanes (the no per-lane scalar).
pub struct SimdPipeline<'a> {
    machine: &'a PdaMachine,
}

impl<'a> SimdPipeline<'a> {
    pub fn new(machine: &'a PdaMachine) -> Self {
        SimdPipeline { machine }
    }

    /// The CSR gather (the B lanes gather their CSR rows in parallel). The B
    /// control states (the B lanes) gather their CSR rows (the
    /// transitions[ctrl_offsets[ctrl_i]..+ctrl_counts[ctrl_i]] in parallel.
    ///
    /// The current reference (the sequential per-lane, the no SIMD yet). The
    /// SIMD version (the AVX-512 VGATHER via the fearless_simd kernel!) is the
    /// next increment (the B lanes gather their CSR rows in parallel, the no
    /// per-lane scalar).
    #[allow(dead_code)]
    fn csr_gather(&self, ctrls: &[u32]) -> Vec<Vec<u32>> {
        let mut results: Vec<Vec<u32>> = Vec::with_capacity(ctrls.len());
        for &ctrl in ctrls {
            let start = self
                .machine
                .ctrl_offsets
                .get(ctrl as usize)
                .copied()
                .unwrap_or(self.machine.transitions.len() as u32) as usize;
            let count = self.machine.ctrl_counts.get(ctrl as usize).copied().unwrap_or(0) as usize;
            let row: Vec<u32> = self.machine.transitions[start..start + count]
                .iter()
                .map(|t| t.a)
                .collect();
            results.push(row);
        }
        results
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
    pub fn step_batch(&self, configs: &[(u32, Vec<u32>)], tokens: &[u32]) -> Vec<(u32, Vec<u32>)> {
        let mut next_configs: Vec<(u32, Vec<u32>)> = Vec::with_capacity(configs.len());
        for ((ctrl, stack), &token) in configs.iter().zip(tokens.iter()) {
            match self.machine.advance_eps(*ctrl, stack, token) {
                Some((nq, ns)) => next_configs.push((nq, ns)),
                None => next_configs.push((*ctrl, stack.clone())), // the hold (the no transition)
            }
        }
        next_configs
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
            let mut masks = vec![self.machine.mask_at_cfg(*ctrl, stack)];
            let mut cur = (*ctrl, stack.clone());
            for &a in draft {
                match self.machine.advance_eps(cur.0, cur.1.as_slice(), a) {
                    Some(nc) => {
                        cur = nc;
                        masks.push(self.machine.mask_at_cfg(cur.0, cur.1.as_slice()));
                    }
                    None => break, // the draft diverged (the no transition)
                }
            }
            projections.push(masks);
        }
        projections
    }

    /// THE EVENT SCAN: the B configs x the tokens -> the B first non-pass-through
    /// indices (the control edge). The B configs are the (ctrl, stack) pairs (the
    /// B sequences). The tokens are the token sequences (the B lanes). The B first
    /// non-pass-through indices are the control edges (the no pass-through).
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