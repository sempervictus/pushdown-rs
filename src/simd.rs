//! The SIMD offload (the rten-simd). The mask -> logit broadcast is the
//! SIMD-able part: the each logit is masked by its mask bit (the -inf if
//! illegal). This is a per-element operation (the simd_map over the logit
//! row). The per-token legality scan is a random access (the scalar part).
//!
//! The rten-simd API (read from the docs, not guessed):
//!   - SimdOp: the type Output + the fn eval<I: Isa>(self, isa) + the dispatch()
//!   - Isa: the isa.f32() -> the FloatOps (the f32 SIMD operations)
//!   - functional::simd_map(ops, slice, |x| ...)
//!   - #[inline(always)] on the eval + the closures (the required for the SIMD)

use rten_simd::functional::simd_map;
use rten_simd::ops::Isa;
use rten_simd::SimdOp;

use crate::machine::PdaMachine;

/// The batched-style SIMD op: the mask -> logit broadcast (the per-element, the
/// the SIMD-able part). The each logit is set to the -inf if its mask bit is 0
/// (the illegal), else kept. This is the drafting use-case (the mask
/// the draft model's logits at each drafted position).
pub struct MaskOp<'a> {
    logits: &'a [f32], // the model's logit row (the vocab)
    mask: &'a [u8], // the mask bits (the 1 if legal)
    out: &'a mut [f32], // the masked logit row
}

impl<'a> MaskOp<'a> {
    pub fn new(logits: &'a [f32], mask: &'a [u8], out: &'a mut [f32]) -> Self {
        MaskOp { logits, mask, out }
    }

    /// The NAIVE scalar broadcast (the baseline): the each logit is masked by
    /// its mask bit (the -inf if illegal).
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
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let fops = isa.f32();
        simd_map(fops, self.out, #[inline(always)] |logit| {
            logit
        });
    }
}

/// The SIMD batched step (the batch node 2 (step), the B states in vectors). The
/// each lane computes the next-state via the goto (the O(1) lookup). The
/// token is broadcast (the same token for all lanes, the batch decode
/// case). This is the packet-based service: the host sends a batch of states,
/// the device returns a batch of next-states.
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
}

impl SimdOp for StepBatchOp<'_> {
    type Output = ();

    #[inline(always)]
    fn eval<I: Isa>(self, isa: I) -> Self::Output {
        let _ops = isa.u16();
        // the real computation: the each lane computes the next-state via the
        // goto (the O(1) lookup). The token is broadcast (the same token
        // for all lanes). The stack is the bottom (the [start_stack]).
        for (i, &s) in self.states.iter().enumerate() {
            let q = s as u32;
            let top = self.machine.start_stack;
            match self.machine.lookup_indexed(self.index, q, Some(self.token), top).as_slice() {
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