//! # pushdown-rs
//!
//! Pushdown automata (PDA) as composable traits, compiled from any
//! finite-state grammar, producing flat bitvec machine encodings.
//!
//! ## The domain
//!
//! A pushdown automaton is the 7-tuple (Q, Sigma, Gamma, delta, q0, Z0, F):
//! the finite control Q, the input alphabet Sigma, the stack alphabet Gamma,
//! the transition relation delta, the start state q0, the start stack symbol
//! Z0, the accepting states F. The variants:
//!
//! - the NPDA (the non-deterministic): the delta is a relation (the multiple
//!   transitions per (q, a, top)); the if ANY computation accepts.
//! - the DPDA (the deterministic): the delta is a function (the at most one
//!   transition per (q, a, top)); accepts if the unique path accepts.
//! - the epsilon-PDA: the delta includes the epsilon-input (the a = None).
//! - the final-state PDA: the acceptance by the F (the final states).
//! - the empty-stack PDA: the acceptance by the empty stack.
//! - the visibly-pushdown automaton (VPA): the input symbols are classified as
//!   the call/return/internal (the VPLs, a subset of the DCFLs).
//! - the alternating PDA: the transitions carry the AND/OR branching.
//! - the one-way stack automaton: the stack head moves only downward.
//! - the nested stack automaton: the stack holds sub-stacks.
//!
//! The [`pda`] module defines the common [`Pda`] trait (the 7-tuple) + the
//! variant traits (the [`Npda`], the [`Dpda`], the [`EpsilonPda`], the
//! [`FinalStatePda`], the [`EmptyStackPda`], the [`VisiblyPushdown`], the
//! [`AlternatingPda`], the [`OneWayStack`], the [`NestedStack`]).
//!
//! ## The compilation
//!
//! The [`compile`] module implements the RTN (recursive-transition-network)
//! compilation of a CFG to a PDA (Alpay & Senturk, arXiv:2603.05540,
//! Definition 5), with the exact control-state count kappa(G) (Definition 10,
//! Lemma 2). The [`Grammar`] trait abstracts the CFG (the N, the Sigma, the P,
//! the S) so any concrete grammar (the Lark, the JSON schema, the BNF, the
//! EBNF) can be compiled to a PDA.
//!
//! ## The machine
//!
//! The [`machine`] module defines the concrete [`PdaMachine`] (the 7-tuple with
//! u32 IDs) + the variant impls (the NPDA simulation, the DPDA simulation, the
//! determinism check). The simulation is bounded by the max_stack (the pushdown
//! depth) + the max_configs (the time safeguard) - the production safeguards.
//!
//! ## The bitvec
//!
//! The [`bitvec`] module provides the flat, uploadable encoding of a machine
//! (the POD table for a GPU/SIMD construction). The encoding is pure POD (no
//! pointers, no Rust-only types) so it can be uploaded to a GPU/SIMD device and
//! the machine reconstructed there.
//!
//! ## Usage
//!
//! ```
//! use pushdown_rs::compile::Cfg;
//! use pushdown_rs::pda::Dpda;
//!
//! // the {a^n b^n} DPDA (the JFLAP tutorial): the push/popush construction
//! const A: u32 = 0;
//! const B: u32 = 1;
//! const EPS: u32 = 2;
//! const Z: u32 = 0;
//! const A_SYM: u32 = 1;
//! let m = pushdown_rs::PdaMachine {
//!     num_states: 4,
//!     num_inputs: 2,
//!     num_stack_syms: 2,
//!     transitions: vec![
//!         pushdown_rs::Transition { q: 0, a: A, top: Z, next_q: 1, push: vec![A_SYM, Z] },
//!         pushdown_rs::Transition { q: 1, a: A, top: A_SYM, next_q: 1, push: vec![A_SYM, A_SYM] },
//!         pushdown_rs::Transition { q: 1, a: B, top: A_SYM, next_q: 2, push: vec![] },
//!         pushdown_rs::Transition { q: 2, a: B, top: A_SYM, next_q: 2, push: vec![] },
//!         pushdown_rs::Transition { q: 2, a: EPS, top: Z, next_q: 3, push: vec![Z] },
//!     ],
//!     accepting: vec![3],
//!     start_state: 0,
//!     start_stack: Z,
//! };
//! assert!(m.is_deterministic());
//! assert!(m.accepts_dpda(&[A, B]));
//! assert!(m.accepts_dpda(&[A, A, B, B]));
//! assert!(!m.accepts_dpda(&[A, A, B]));
//! ```

pub mod pda;
pub mod machine;
pub mod compile;
pub mod bitvec;
#[cfg(feature = "simd")]
pub mod simd;
pub mod summary;
pub mod mask_class;
pub mod oracle;
pub mod graph;
pub mod cuda;
pub mod spanner;
pub mod service;

pub use pda::{
    AlternatingPda, Dpda, EmptyStackPda, EpsilonPda, FinalStatePda, NestedStack, OneWayStack, Npda,
    Pda, VisiblyPushdown,
};
pub use machine::{PdaError, PdaMachine, Transition};
pub use compile::{Cfg, CfgError, Grammar, kappa};
pub use bitvec::BitvecError;

/// Compile an abstract grammar to a PDA machine (the RTN construction).
/// Proves the input is a valid CFG (the validate) before compiling.
pub fn compile<G: Grammar>(g: &G) -> Result<PdaMachine, CfgError> {
    compile::compile(g)
}