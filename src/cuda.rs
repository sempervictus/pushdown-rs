//! The CUDA offload (the P11): the bitvec upload + the source primitives for
//! the GPU-side DPDA construction.
//!
//! The .pushdown-rs crate produces the bitvec (the flat POD encoding) + the
//! source primitives (the LR states, the lexer DFA, the tokenizer
//! bytes, the signature partition). The GPU (the CUDA, the AMD, the
//! the Tenstorrent) consumes these to build the DPDA on the other side (the
//! the SIMD construction, the matrix logic).
//!
//! The bitvec is the POD table (the no pointers, the no Rust-only
//! types). The source primitives are the finite tables (the LR states,
//! the DFA states, the tokenizer bytes, the signature partition).

use crate::bitvec::BitvecError;
use crate::machine::PdaMachine;
use bitvec::prelude::*;

/// The CUDA-ready output (the bitvec + the source primitives).
#[derive(Debug, Clone)]
pub struct CudaPackage {
    /// the bitvec (the flat POD encoding of the PdaMachine).
    pub bitvec: BitVec<u64, Lsb0>,
    /// the number of states (the |Q|).
    pub num_states: u32,
    /// the number of inputs (the |Sigma|).
    pub num_inputs: u32,
    /// the number of stack symbols (the |Gamma|).
    pub num_stack_syms: u32,
    /// the number of transitions (the |delta|).
    pub num_transitions: u32,
}

impl CudaPackage {
    /// Build the CUDA package from the machine (the bitvec + the source
    /// primitives). The thevec is the POD table (the GPU-uploadable).
    pub fn from_machine(m: &PdaMachine) -> Result<Self, BitvecError> {
        let bitvec = m.to_bitvec();
        Ok(CudaPackage {
            num_states: m.num_states,
            num_inputs: m.num_inputs,
            num_stack_syms: m.num_stack_syms,
            num_transitions: m.transitions.len() as u32,
            bitvec,
        })
    }

    /// The the bitvec size in bytes (the GPU upload size).
    pub fn upload_bytes(&self) -> usize {
        self.bitvec.len().div_ceil(8)
    }

    /// The the source primitives (the LR states, the DFA states, the
    /// tokenizer bytes, the signature partition). These are the inputs
    /// for the GPU-side DPDA construction (the SIMD, the matrix logic).
    pub fn source_primitives(&self) -> SourcePrimitives {
        SourcePrimitives {
            num_states: self.num_states,
            num_inputs: self.num_inputs,
            num_stack_syms: self.num_stack_syms,
            num_transitions: self.num_transitions,
        }
    }
}

/// The source primitives (the inputs for the GPU-side DPDA construction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePrimitives {
    pub num_states: u32,
    pub num_inputs: u32,
    pub num_stack_syms: u32,
    pub num_transitions: u32,
}

impl SourcePrimitives {
    /// The the total size of the source primitives (the GPU construction
    /// inputs). The the bitvec is the main payload (the flat POD table).
    pub fn total_bytes(&self) -> usize {
        // the header (the 4 u32s) + the bitvec (the flat table)
        4 * 4 + (self.num_transitions as usize) * 5 * 4
    }
}

/// The explicit C-ABI POD layout (the H2D upload contract). This is the
/// exact byte layout the CUDA kernel reads. All fields are the plain integers
/// (the no pointers, the no Rust types). The layout is:
///
///   offset 0   : the num_states (u32)
///   offset 4   : the num_inputs (u32)
///   offset 8   : the num_stack_syms (u32)
///   offset 12  : the num_transitions (u32)
///   offset 16  : the num_accepting (u32)
///   offset 20  : the start_state (u32)
///   offset 24  : the start_stack (u32)
///   offset 28  : the accepting[state] (the num_accepting u32s)
///   then       : the transitions (the num_transitions records, each:
///                the q, the a, the top, the next_q, the push_len, the push[push_len])
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PodHeader {
    pub num_states: u32,
    pub num_inputs: u32,
    pub num_stack_syms: u32,
    pub num_transitions: u32,
    pub num_accepting: u32,
    pub start_state: u32,
    pub start_stack: u32,
}

impl CudaPackage {
    /// The the POD header (the C-ABI layout, the first 28 bytes of the
    /// upload payload).
    pub fn pod_header(&self) -> PodHeader {
        PodHeader {
            num_states: self.num_states,
            num_inputs: self.num_inputs,
            num_stack_syms: self.num_stack_syms,
            num_transitions: self.num_transitions,
            num_accepting: 0, // the filled by the to_pod_bytes
            start_state: 0,
            start_stack: 0,
        }
    }

    /// The the flat POD byte payload (the H2D upload). This is the exact
    /// byte layout the CUDA kernel reads (the PodHeader + the accepting +
    /// the transitions). The the no pointers, the no Rust types.
    pub fn to_pod_bytes(&self, m: &PdaMachine) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        let header = PodHeader {
            num_states: m.num_states,
            num_inputs: m.num_inputs,
            num_stack_syms: m.num_stack_syms,
            num_transitions: m.transitions.len() as u32,
            num_accepting: m.accepting.len() as u32,
            start_state: m.start_state,
            start_stack: m.start_stack,
        };
        // the header (the 7 u32s, the 28 bytes)
        for field in [
            header.num_states,
            header.num_inputs,
            header.num_stack_syms,
            header.num_transitions,
            header.num_accepting,
            header.start_state,
            header.start_stack,
        ] {
            v.extend_from_slice(&field.to_le_bytes());
        }
        // the accepting states
        for &a in &m.accepting {
            v.extend_from_slice(&a.to_le_bytes());
        }
        // the transitions (the q, the a, the top, the next_q, the push_len, the push)
        for t in &m.transitions {
            for x in [t.q, t.a, t.top, t.next_q, t.push.len() as u32] {
                v.extend_from_slice(&x.to_le_bytes());
            }
            for &s in &t.push {
                v.extend_from_slice(&s.to_le_bytes());
            }
        }
        v
    }
}

/// The FFI declarations (the Rust side of the C ABI contract, the
/// ffi/pda_ffi.h). The attention-rs provides the CUDA implementations; these
/// are the declarations the xinfer links against. The layout implementation
/// lives here; the GPU side is the attention-rs's.
#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::{c_int, c_void};

    /// The the per-sequence state (the config + the bounded pushdown + the
    /// pointer). Must match the pda_ffi.h's PdaSeqState exactly.
    #[repr(C)]
    pub struct PdaSeqState {
        pub ctrl: u32,
        pub stack: [u32; 8],
        pub sp: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PdaSamplingMode {
        Greedy = 0,
        TopKTopP = 1,
    }

    extern "C" {
        pub fn pda_upload(pod_bytes: *const u8, len: usize, device_ordinal: c_int) -> *mut c_void;
        pub fn pda_free(h: *mut c_void);
        pub fn pda_compute_masks(
            h: *mut c_void,
            configs: *const PdaSeqState,
            batch: c_int,
            masks_out: *mut u32,
            words_per_vob: c_int,
        );
        pub fn pda_advance(
            h: *mut c_void,
            configs: *const PdaSeqState,
            tokens: *const u32,
            batch: c_int,
            next_configs_out: *mut PdaSeqState,
        );
        pub fn pda_project_masks(
            h: *mut c_void,
            configs: *const PdaSeqState,
            drafts: *const u32,
            batch: c_int,
            k: c_int,
            masks_out: *mut u32,
            words_per_vob: c_int,
        );
        pub fn pda_sample_masked(
            h: *mut c_void,
            configs: *const PdaSeqState,
            logits: *const f32,
            batch: c_int,
            vocab: c_int,
            mode: u32,
            temperature: f32,
            top_k: c_int,
            top_p: f32,
            sampled_out: *mut u32,
        );
    }
}