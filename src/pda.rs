//! The pushdown-automaton domain, as composable traits.
//!
//! The common [`Pda`] trait is the 7-tuple (Q, Sigma, Gamma, delta, q0, Z0, F).
//! The specific variants layer on top by trait composition:
//!
//!   - [`Npda`]          non-deterministic (accept path accepts)
//!   - [`Dpda`]          deterministic (at most one transition per (q, a, top))
//!   - [`EpsilonPda`]    epsilon-transitions
//!   - [`FinalStatePda`] acceptance by final state
//!   - [`EmptyStackPda`] acceptance by empty stack
//!   - [`VisiblyPushdown`] visibly-pushdown (call/return/internal)
//!   - [`AlternatingPda`]  alternating (AND/OR)
//!   - [`OneWayStack`]     one-way stack automaton
//!   - [`NestedStack`]     nested stack automaton
//!
//! A concrete machine implements [`Pda`] plus the variant traits it satisfies.

use std::hash::Hash;

/// The common pushdown automaton (the 7-tuple Q, Sigma, Gamma, delta, q0, Z0, F).
///
/// The transition `delta(q, a, top)` yields the set of (q', gamma') pairs. For a
/// deterministic machine this set has at most one element; for a non-deterministic
/// machine it may have several. `a` is `None` for the epsilon-input.
pub trait Pda {
    /// the finite control states (Q).
    type State: Copy + Eq + Hash;
    /// the input alphabet (Sigma).
    type Input: Copy + Eq + Hash;
    /// the stack alphabet (Gamma).
    type StackSym: Copy + Eq + Hash;
    /// the push string (the Gamma*).
    type Push: AsRef<[Self::StackSym]>;

    fn states(&self) -> Vec<Self::State>;
    fn inputs(&self) -> Vec<Self::Input>;
    fn stack_syms(&self) -> Vec<Self::StackSym>;
    fn start_state(&self) -> Self::State;
    fn start_stack(&self) -> Self::StackSym;
    fn accepting(&self) -> Vec<Self::State>;

    /// The transition delta(q, a, top) -> the set of (q', gamma').
    fn transition(
        &self,
        q: Self::State,
        a: Option<Self::Input>,
        top: Self::StackSym,
    ) -> Vec<(Self::State, Self::Push)>;
}

/// A non-deterministic pushdown automaton (NPDA): accepts if ANY computation
/// accepts. `max_stack` bounds the pushdown depth; `max_configs` bounds the
/// search (the production time safeguard - the BFS frontier cap).
pub trait Npda: Pda {
    fn accepts_npda(&self, w: &[Self::Input], max_stack: usize, max_configs: usize) -> bool;
}

/// A deterministic pushdown automaton (DPDA): at most one transition per
/// (q, a, top). Accepts if the unique path reaches an accepting configuration.
pub trait Dpda: Pda {
    /// True iff the machine is deterministic (no (q, a, top) has two transitions).
    fn is_deterministic(&self) -> bool;
    fn accepts_dpda(&self, w: &[Self::Input]) -> bool;
}

/// The streaming PDA interface (the batched pipeline node, the batched pipeline).
/// The PRIMARY interface when the simd feature is on (the batched step +
/// the projection). The scalar code is the fallback (the no simd feature).
///
/// The batched invariant: the batched op over a batch == the scalar op applied
/// per-item (the differential).
pub trait PdaStream: Pda {
    /// The the batch of configs (the (state, stack) pairs).
    type Config;
    /// The the mask (the set of legal inputs).
    type Mask;
    /// The the batched step (the batched Node 2): the B (config, token) -> the B next-config.
    fn step_batch(&self, batch: &[(Self::Config, Self::Input)]) -> Vec<Self::Config>;
    /// The the batched mask (the batched Node 1): the B configs -> the B masks.
    fn mask_batch(&self, configs: &[Self::Config]) -> Vec<Self::Mask>;
    /// The the batched projection (the batched Node 3): the B configs x the K
    /// drafts -> the B x (K+1) masks (the drafting use-case).
    fn project_batch(&self, configs: &[Self::Config], drafts: &[Vec<Self::Input>]) -> Vec<Vec<Self::Mask>>;
}

/// An epsilon-PDA: the transition relation includes the epsilon-input (a = None).
pub trait EpsilonPda: Pda {}

/// A final-state PDA: acceptance is by reaching a state in F with the input exhausted.
pub trait FinalStatePda: Pda {}

/// An empty-stack PDA: acceptance is by emptying the stack (the input exhausted).
pub trait EmptyStackPda: Pda {}

/// A visibly-pushdown automaton (VPA): each input symbol is classified as a
/// call, a return, or an internal symbol (the VPLs, a subset of the DCFLs).
pub trait VisiblyPushdown: Pda {
    fn is_call(&self, a: Self::Input) -> bool;
    fn is_return(&self, a: Self::Input) -> bool;
    fn is_internal(&self, a: Self::Input) -> bool;
}

/// An alternating pushdown automaton: the transitions carry AND/OR branching
/// (the alternating, strictly more expressive than the NPDA).
pub trait AlternatingPda: Pda {}

/// A one-way stack automaton: the stack head moves only downward (no re-reading).
pub trait OneWayStack: Pda {}

/// A nested stack automaton: the stack holds sub-stacks (the nested, strictly
/// more expressive than the PDA).
pub trait NestedStack: Pda {}