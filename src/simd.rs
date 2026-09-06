//! The SIMD offload (the rten-simd). The mask -> logit broadcast is the
//! SIMD-able part: each logit is masked by its mask bit (-inf if illegal).
//!
//! The rten-simd API (verified from docs.rs/rten-simd/0.26.0):
//!   - SimdOp: type Output + fn eval<I: Isa>(self, isa: I) -> Self::Output + dispatch()
//!   - Isa: f.f32() -> impl FloatOps<f32> (extends NumOps, extends BitOps)
//!   - BitOps: load(&[f32]) -> Simd, store(Simd, &mut [f32]), len() -> usize,
//!             splat(f32) -> Simd, select(x, y, mask) -> Simd
//!   - NumOps: add(x, y) -> Simd, mul(x, y) -> Simd, sub(x, y) -> Simd
//!   - Lane widths: GenericIs4, AVX2=8, AVX512=16 (for f32)
//!
//! The SIMD win: the vocab is 248K f32 elements. The mask application is a
//! vectorized add across the whole vocab (logits + bias where bias is 0.0 or
//! -inf). This is O(vocab/lane) SIMD iterations vs O(vocab) scalar.

use rten_simd::ops::{BitOps, Isa, NumOps};
use rten_simd::SimdOp;

use crate::machine::PdaMachine;

/// Precompute the f32 bias array from a u32 VOB mask.
/// bias[i] = 0.0 if mask bit i is set (allowed), -inf if clear (disallowed).
/// This is a one-time scalar cost per PDA state.
pub fn compute_bias(mask_words: &[u32], vocab_size: usize) -> Vec<f32> {
    let mut bias = vec![f32::NEG_INFINITY; vocab_size];
    for (w, &word) in mask_words.iter().enumerate() {
        for bit in 0..32 {
            let pos = w * 32 + bit;
            if pos >= vocab_size {
                break;
            }
            if (word >> bit) & 1 == 1 {
                bias[pos] = 0.0;
            }
        }
    }
    bias
}

/// The mask broadcast op: apply a precomputed f32 bias to a logit row.
/// out[i] = logits[i] + bias[i]. Where bias is -inf, the result is -inf.
/// This is the primary SIMD win: vectorized add across the vocab.
pub struct MaskBroadcastOp<'a> {
    logits: &'a [f32],
    bias: &'a [f32],
    out: &'a mut [f32],
}

impl<'a> MaskBroadcastOp<'a> {
    pub fn new(logits: &'a [f32], bias: &'a [f32], out: &'a mut [f32]) -> Self {
        MaskBroadcastOp { logits, bias, out }
    }

    /// Scalar reference (always correct, used for the tail and for proof).
    pub fn scalar(&mut self) {
        for i in 0..self.out.len() {
            self.out[i] = self.logits[i] + self.bias[i];
        }
    }
}

impl SimdOp for MaskBroadcastOp<'_> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let fops = isa.f32();
        let lane = fops.len();
        let n = self.out.len();
        let mut i = 0usize;

        // Vectorized body: process `lane` f32 elements at a time.
        while i + lane <= n {
            let v = fops.load(&self.logits[i..i + lane]);
            let b = fops.load(&self.bias[i..i + lane]);
            let result = fops.add(v, b);
            fops.store(result, &mut self.out[i..i + lane]);
            i += lane;
        }

        // Scalar tail (fewer than `lane` elements remaining).
        while i < n {
            self.out[i] = self.logits[i] + self.bias[i];
            i += 1;
        }
    }
}

/// The batched step op: advance B PDA states by a broadcast token.
/// Each lane holds one state; the token is the same for all lanes (batch decode).
///
/// The transition lookup is a random-access HashMap (scalar). The SIMD win
/// here is limited to the u16 state load/store. For future, the step is
/// scalar per-lane; the batch dimension is what the CUDA kernel vectorizes.
/// This op exists for the batch-invariant proof and for future table-based
/// (non-HashMap) lookups.
pub struct StepBatchOp<'a> {
    machine: &'a PdaMachine,
    index: &'a std::collections::HashMap<(u32, u32, u32), Vec<usize>>,
    states: &'a [u16],
    token: u32,
    out: &'a mut [u16],
}

impl<'a> StepBatchOp<'a> {
    pub fn new(
        machine: &'a PdaMachine,
        index: &'a std::collections::HashMap<(u32, u32, u32), Vec<usize>>,
        states: &'a [u16],
        token: u32,
        out: &'a mut [u16],
    ) -> Self {
        StepBatchOp { machine, index, states, token, out }
    }

    /// Scalar reference implementation.
    pub fn scalar(&mut self) {
        for (i, &s) in self.states.iter().enumerate() {
            let q = s as u32;
            let top = self.machine.start_stack;
            match self
                .machine
                .lookup_indexed(self.index, q, Some(self.token), top)
                .as_slice()
            {
                [t] => {
                    self.out[i] = t.next_q as u16;
                }
                _ => {
                    self.out[i] = q as u16;
                }
            }
        }
    }
}

impl SimdOp for StepBatchOp<'_> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        // The step is a random-access lookup (HashMap), which does not
        // vectorize. Use the ISA for the u16 load/store of the state array
        // (the batch dimension). The actual transition resolution is scalar
        // per-lane.
        let u16_ops = isa.u16();
        let lane = u16_ops.len();
        let n = self.out.len();
        let mut i = 0usize;

        while i + lane <= n {
            // Load `lane` states as a SIMD vector.
            let v = u16_ops.load(&self.states[i..i + lane]);
            // Store the same values to out (identity for states that have
            // no transition; the actual advance-state happens per-lane below).
            u16_ops.store(v, &mut self.out[i..i + lane]);
            // Per-lane scalar resolution (the HashMap lookup cannot be SIMD).
            for j in 0..lane {
                let q = self.states[i + j] as u32;
                let top = self.machine.start_stack;
                if let [t] = self
                    .machine
                    .lookup_indexed(self.index, q, Some(self.token), top)
                    .as_slice()
                {
                    self.out[i + j] = t.next_q as u16;
                }
            }
            i += lane;
        }

        // Scalar tail.
        while i < n {
            let q = self.states[i] as u32;
            let top = self.machine.start_stack;
            match self
                .machine
                .lookup_indexed(self.index, q, Some(self.token), top)
                .as_slice()
            {
                [t] => {
                    self.out[i] = t.next_q as u16;
                }
                _ => {
                    self.out[i] = self.states[i];
                }
            }
            i += 1;
        }
    }
}

/// The old MaskOp (kept for API compatibility with existing tests).
/// Delegates to MaskBroadcastOp internally.
pub struct MaskOp<'a> {
    logits: &'a [f32],
    mask: &'a [u8],
    out: &'a mut [f32],
}

impl<'a> MaskOp<'a> {
    pub fn new(logits: &'a [f32], mask: &'a [u8], out: &'a mut [f32]) -> Self {
        MaskOp { logits, mask, out }
    }

    /// The NAIVE scalar broadcast (the baseline).
    pub fn scalar(&mut self) {
        for i in 0..self.out.len() {
            let legal = self.mask.get(i).copied().unwrap_or(0) == 1;
            let logit = *self.logits.get(i).unwrap_or(&0.0);
            self.out[i] = if legal { logit } else { f32::NEG_INFINITY };
        }
    }
}

impl SimdOp for MaskOp<'_> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, _isa: I) -> Self::Output {
        // Convert the u8 mask to a u32 VOB, then use the real SIMD broadcast.
        let n = self.out.len();
        let num_words = (n + 31) / 32;
        let mut vob = vec![0u32; num_words];
        for i in 0..n {
            if self.mask.get(i).copied().unwrap_or(0) == 1 {
                vob[i / 32] |= 1u32 << (i % 32);
            }
        }
        let bias = compute_bias(&vob, n);
        let op = MaskBroadcastOp::new(self.logits, &bias, self.out);
        op.dispatch();
    }
}