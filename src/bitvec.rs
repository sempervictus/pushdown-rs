//! The bitvec encoding of a PDA machine (the flat, uploadable table).
//!
//! The encoding is pure POD (no pointers, no Rust-only types) so it can be
//! uploaded to a GPU/SIMD device and the machine reconstructed there. The
//! `bitvec` crate provides the bit-level storage.

use bitvec::prelude::*;
use crate::machine::{PdaMachine, Transition};

/// Errors from the bitvec (de)serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitvecError {
    /// The bitvec is malformed (truncated, bad length, out-of-range index).
    Malformed(String),
}
impl std::fmt::Display for BitvecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitvecError::Malformed(msg) => write!(f, "malformed PDA bitvec: {msg}"),
        }
    }
}
impl std::error::Error for BitvecError {}

type Bits = BitVec<u64, Lsb0>;

fn push_u32(bits: &mut Bits, v: u32) {
    for i in (0..32).rev() {
        bits.push(((v >> i) & 1) == 1);
    }
}
fn load_u32(bits: &BitSlice<u64, Lsb0>, off: &mut usize) -> Result<u32, BitvecError> {
    if *off + 32 > bits.len() {
        return Err(BitvecError::Malformed("truncated u32".into()));
    }
    let mut v = 0u32;
    for _ in 0..32 {
        v = (v << 1) | (bits[*off] as u32);
        *off += 1;
    }
    Ok(v)
}

impl PdaMachine {
    /// Serialize the machine to a flat bitvec (the POD table).
    ///
    /// Layout (all big-endian u32):
    ///   num_states, num_inputs, num_stack_syms, num_transitions, num_accepting,
    ///   start_state, start_stack, the accepting IDs, then per transition:
    ///   q, a, top, next_q, push_len, push[0..push_len).
    pub fn to_bitvec(&self) -> Bits {
        let mut bits = Bits::new();
        for &x in &[
            self.num_states,
            self.num_inputs,
            self.num_stack_syms,
            self.transitions.len() as u32,
            self.accepting.len() as u32,
            self.start_state,
            self.start_stack,
        ] {
            push_u32(&mut bits, x);
        }
        for &a in &self.accepting {
            push_u32(&mut bits, a);
        }
        for t in &self.transitions {
            push_u32(&mut bits, t.q);
            push_u32(&mut bits, t.a);
            push_u32(&mut bits, t.top);
            push_u32(&mut bits, t.next_q);
            push_u32(&mut bits, t.push.len() as u32);
            for &s in &t.push {
                push_u32(&mut bits, s);
            }
        }
        bits
    }

    /// Deserialize a machine from a flat bitvec. Validates the bounds.
    pub fn from_bitvec(bits: &bitvec::slice::BitSlice<u64, Lsb0>) -> Result<Self, BitvecError> {
        let mut off = 0usize;
        let num_states = load_u32(bits, &mut off)?;
        let num_inputs = load_u32(bits, &mut off)?;
        let num_stack_syms = load_u32(bits, &mut off)?;
        let num_transitions = load_u32(bits, &mut off)?;
        let num_accepting = load_u32(bits, &mut off)?;
        let start_state = load_u32(bits, &mut off)?;
        let start_stack = load_u32(bits, &mut off)?;
        let mut accepting = Vec::with_capacity(num_accepting as usize);
        for _ in 0..num_accepting {
            accepting.push(load_u32(bits, &mut off)?);
        }
        let mut transitions = Vec::with_capacity(num_transitions as usize);
        for _ in 0..num_transitions {
            let q = load_u32(bits, &mut off)?;
            let a = load_u32(bits, &mut off)?;
            let top = load_u32(bits, &mut off)?;
            let next_q = load_u32(bits, &mut off)?;
            let push_len = load_u32(bits, &mut off)?;
            let mut push = Vec::with_capacity(push_len as usize);
            for _ in 0..push_len {
                push.push(load_u32(bits, &mut off)?);
            }
            transitions.push(Transition {
                q,
                a,
                top,
                next_q,
                push,
            });
        }
        let m = PdaMachine {
            num_states,
            num_inputs,
            num_stack_syms,
            transitions,
            accepting,
            start_state,
            start_stack,
        };
        m.validate_bounds().map_err(|e| BitvecError::Malformed(e.to_string()))?;
        Ok(m)
    }
}