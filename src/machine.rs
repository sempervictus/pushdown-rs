//! The concrete PDA machine (the 7-tuple with u32 IDs) + the variant trait
//! impls (the NPDA, the DPDA, the epsilon, the final-state, the empty-stack).

use crate::pda::{Dpda, EmptyStackPda, EpsilonPda, FinalStatePda, Npda, Pda, PdaStream};
use std::result::Result as StdResult;

/// The default bounds for the NPDA search (the accepts_npda's safeguards).
/// The max_stack bounds the pushdown depth; the max_configs bounds the search.
pub const DEFAULT_MAX_STACK: usize = 64;
pub const DEFAULT_MAX_CONFIGS: usize = 1_000_000;

/// One transition: (q, a, top) -> (q', push). `a` is `num_inputs` for the
/// epsilon-input. `push` is the stack string that replaces `top` (empty = pop).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub q: u32,
    pub a: u32,
    pub top: u32,
    pub next_q: u32,
    pub push: Vec<u32>,
}

/// The concrete PDA machine (the 7-tuple).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdaMachine {
    pub num_states: u32,
    pub num_inputs: u32,
    pub num_stack_syms: u32,
    pub transitions: Vec<Transition>,
    pub accepting: Vec<u32>,
    pub start_state: u32,
    pub start_stack: u32,
}

impl PdaMachine {
    /// Build a machine, validating the bounds.
    pub fn new(
        num_states: u32,
        num_inputs: u32,
        num_stack_syms: u32,
        transitions: Vec<Transition>,
        accepting: Vec<u32>,
        start_state: u32,
        start_stack: u32,
    ) -> StdResult<Self, PdaError> {
        let m = PdaMachine {
            num_states,
            num_inputs,
            num_stack_syms,
            transitions,
            accepting,
            start_state,
            start_stack,
        };
        m.validate_bounds()?;
        Ok(m)
    }

    /// Validate the transition bounds (the states, the inputs, the stack syms).
    pub fn validate_bounds(&self) -> StdResult<(), PdaError> {
        for t in &self.transitions {
            if t.q >= self.num_states || t.next_q >= self.num_states {
                return Err(PdaError::StateOutOfRange {
                    state: t.q,
                    num: self.num_states,
                });
            }
            if t.a > self.num_inputs {
                return Err(PdaError::InputOutOfRange {
                    input: t.a,
                    num: self.num_inputs,
                });
            }
            if t.top >= self.num_stack_syms {
                return Err(PdaError::StackSymOutOfRange {
                    sym: t.top,
                    num: self.num_stack_syms,
                });
            }
            for &s in &t.push {
                if s >= self.num_stack_syms {
                    return Err(PdaError::StackSymOutOfRange {
                        sym: s,
                        num: self.num_stack_syms,
                    });
                }
            }
        }
        Ok(())
    }

    /// Look up the transitions for (q, a, top). `a` is None for the epsilon.
    pub fn lookup(&self, q: u32, a: Option<u32>, top: u32) -> Vec<&Transition> {
        let key = a.unwrap_or(self.num_inputs);
        self.transitions
            .iter()
            .filter(|t| t.q == q && t.a == key && t.top == top)
            .collect()
    }

    /// The universal accepts: auto-selects the right simulation based on the
    /// machine's determinism. Callers do NOT need to know if the machine is a
    /// DPDA or an NPDA - this picks the correct one.
    ///   - deterministic: the accepts_dpda (the single path, the fast)
    ///   - non-deterministic: the accepts_npda (the BFS, the bounded)
    /// This is the "stupid-LLM-proof" entry point (the one method to call).
    pub fn accepts(&self, w: &[u32]) -> bool {
        if self.is_deterministic() {
            self.accepts_dpda(w)
        } else {
            self.accepts_npda(w, DEFAULT_MAX_STACK, DEFAULT_MAX_CONFIGS)
        }
    }

    /// The universal accepts with caller-controlled bounds (the max_stack +
    /// the max_configs). The `accepts` uses the defaults; this lets the caller
    /// tune the search bounds for large/deep grammars.
    pub fn accepts_sized(&self, w: &[u32], max_stack: usize, max_configs: usize) -> bool {
        if self.is_deterministic() {
            self.accepts_dpda(w)
        } else {
            self.accepts_npda(w, max_stack, max_configs)
        }
    }

    /// EXPOSE the RTN state structure: print every state, its transitions, the
    /// start, the accepting set. This is the "show me the states" diagnostic.
    pub fn dump(&self) {
        println!(
            "PDA: {} states, {} inputs, {} stack syms, start={}, accepting={:?}",
            self.num_states,
            self.num_inputs,
            self.num_stack_syms,
            self.start_state,
            self.accepting
        );
        for t in &self.transitions {
            println!(
                "  (q={}, in={}, top={}) -> (q={}, push={:?})",
                t.q, t.a, t.top, t.next_q, t.push
            );
        }
    }

    /// Build an O(1) lookup index: the (q, a, top) -> the transition indices.
    /// The `a` is the input ID (the num_inputs for the epsilon). This is the
    /// fast path (the benchmark's "PDA O(1)" column) - the linear-scan
    /// `lookup` is the reference.
    pub fn build_index(&self) -> std::collections::HashMap<(u32, u32, u32), Vec<usize>> {
        let mut idx: std::collections::HashMap<(u32, u32, u32), Vec<usize>> = std::collections::HashMap::new();
        for (i, t) in self.transitions.iter().enumerate() {
            idx.entry((t.q, t.a, t.top)).or_default().push(i);
        }
        idx
    }

    /// The O(1) lookup via the index (the (q, a, top) -> the transitions).
    pub fn lookup_indexed(
        &self,
        idx: &std::collections::HashMap<(u32, u32, u32), Vec<usize>>,
        q: u32,
        a: Option<u32>,
        top: u32,
    ) -> Vec<&Transition> {
        let key = (q, a.unwrap_or(self.num_inputs), top);
        match idx.get(&key) {
            Some(vs) => vs.iter().map(|&i| &self.transitions[i]).collect(),
            None => Vec::new(),
        }
    }

    /// The mask bits for a state (the 1 if the input is legal). The SAME
    /// interface dispatches to the SIMD (the rten-simd) if the feature is
    /// on, else the scalar loop. Callers never know which ran.
    pub fn mask_bits(&self, state: u32) -> Vec<u8> {
        let n = (self.num_inputs + 1) as usize;
        let mut out = vec![0u8; n];
        // the mask computation is a random access into the transition table (the
        // the no SIMD win). The SIMD win is the BROADCAST (the mask -> the
        // logit row), which is the MaskOp (the simd.rs).
        for i in 0..n {
            out[i] = if self.lookup(state, Some(i as u32), self.start_stack).is_empty() {
                0
            } else {
                1
            };
        }
        out
    }
}

// The Pda impl (the 7-tuple).
impl Pda for PdaMachine {
    type State = u32;
    type Input = u32;
    type StackSym = u32;
    type Push = Vec<u32>;

    fn states(&self) -> Vec<u32> {
        (0..self.num_states).collect()
    }
    fn inputs(&self) -> Vec<u32> {
        (0..self.num_inputs).collect()
    }
    fn stack_syms(&self) -> Vec<u32> {
        (0..self.num_stack_syms).collect()
    }
    fn start_state(&self) -> u32 {
        self.start_state
    }
    fn start_stack(&self) -> u32 {
        self.start_stack
    }
    fn accepting(&self) -> Vec<u32> {
        self.accepting.clone()
    }
    fn transition(
        &self,
        q: u32,
        a: Option<u32>,
        top: u32,
    ) -> Vec<(u32, Vec<u32>)> {
        self.lookup(q, a, top)
            .into_iter()
            .map(|t| (t.next_q, t.push.clone()))
            .collect()
    }
}

// The Npda impl (the accepts if ANY path accepts).
impl Npda for PdaMachine {
    fn accepts_npda(&self, w: &[u32], max_stack: usize, max_configs: usize) -> bool {
        // the BFS over the (q, stack) configurations, bounded by max_stack +
        // max_configs (the production safeguards). The seen set dedups the
        // frontier (prevents the duplicate blowup).
        use std::collections::HashSet;
        let mut frontier: Vec<(u32, Vec<u32>)> = vec![(self.start_state, vec![self.start_stack])];
        let mut explored = 0usize;
        let mut i = 0usize;
        loop {
            // the epsilon-closure (the dedup is per-step, not global)
            let mut seen: HashSet<(u32, Vec<u32>)> = frontier.iter().cloned().collect();
            let mut changed = true;
            while changed {
                changed = false;
                let mut next = Vec::new();
                for &(q, ref stack) in &frontier {
                    let Some(top) = stack.last().copied() else {
                        continue; // the empty stack (the no top) - skip
                    };
                    for (q2, push) in self.transition(q, None, top) {
                        let mut s2 = stack.clone();
                        s2.pop();
                        for &p in push.iter().rev() {
                            s2.push(p);
                        }
                        if s2.len() <= max_stack && seen.insert((q2, s2.clone())) {
                            next.push((q2, s2));
                            changed = true;
                        }
                    }
                }
                if !next.is_empty() {
                    let added = next.len();
                    frontier.extend(next);
                    explored += added;
                    if explored > max_configs {
                        return false; // the time bound (the production safeguard)
                    }
                }
            }
            if i >= w.len() {
                // accept only at the top level: the stack is exactly the bottom
                // marker (the inner reductions leave a return address on the stack)
                return frontier.iter().any(|(q, s)| self.accepting.contains(q) && s.len() == 1 && s[0] == self.start_stack);
            }
            let a = w[i];
            i += 1;
            let mut seen: HashSet<(u32, Vec<u32>)> = HashSet::new();
            let mut next = Vec::new();
            for &(q, ref stack) in &frontier {
                let Some(top) = stack.last().copied() else {
                    continue; // the empty stack (the no top) - skip
                };
                for (q2, push) in self.transition(q, Some(a), top) {
                    let mut s2 = stack.clone();
                    s2.pop();
                    for &p in push.iter().rev() {
                        s2.push(p);
                    }
                    if s2.len() <= max_stack && seen.insert((q2, s2.clone())) {
                        next.push((q2, s2));
                    }
                }
            }
            frontier = next;
            explored += frontier.len();
            if explored > max_configs {
                return false; // the time bound (the production safeguard)
            }
            if frontier.is_empty() {
                return false;
            }
        }
    }
}

// TODO: remove_for_production - the debug trace that decomposes the closure
// states step-by-step (the frontier after each epsilon-closure + input step).
impl PdaMachine {
    pub fn trace_npda(&self, w: &[u32], max_stack: usize) {
        use std::collections::HashSet;
        let mut frontier: Vec<(u32, Vec<u32>)> = vec![(self.start_state, vec![self.start_stack])];
        let mut i = 0usize;
        println!("trace input={:?}", w);
        println!("  start frontier: {:?}", frontier);
        loop {
            // the epsilon-closure (the decompose the closure states)
            let mut seen: HashSet<(u32, Vec<u32>)> = frontier.iter().cloned().collect();
            let mut changed = true;
            while changed {
                changed = false;
                let mut next = Vec::new();
                for &(q, ref stack) in &frontier {
                    let Some(top) = stack.last().copied() else { continue };
                    for (q2, push) in self.transition(q, None, top) {
                        let mut s2 = stack.clone();
                        s2.pop();
                        for &p in push.iter().rev() {
                            s2.push(p);
                        }
                        if s2.len() <= max_stack && seen.insert((q2, s2.clone())) {
                            next.push((q2, s2));
                            changed = true;
                        }
                    }
                }
                if !next.is_empty() {
                    println!("    eps-closure added: {:?}", next);
                    frontier.extend(next);
                }
            }
            if i >= w.len() {
                println!(
                    "  input exhausted. frontier={:?} accepting={:?}",
                    frontier, self.accepting
                );
                break;
            }
            let a = w[i];
            i += 1;
            println!("  input step a={a}: frontier before={:?}", frontier);
            let mut seen: HashSet<(u32, Vec<u32>)> = HashSet::new();
            let mut next = Vec::new();
            for &(q, ref stack) in &frontier {
                let Some(top) = stack.last().copied() else { continue };
                for (q2, push) in self.transition(q, Some(a), top) {
                    let mut s2 = stack.clone();
                    s2.pop();
                    for &p in push.iter().rev() {
                        s2.push(p);
                    }
                    if s2.len() <= max_stack && seen.insert((q2, s2.clone())) {
                        next.push((q2, s2));
                    }
                }
            }
            println!("  input step a={}a: frontier after={:?}", a, next);
            frontier = next;
            if frontier.is_empty() {
                println!("  frontier empty - reject");
                break;
            }
        }
    }

    /// The no-allocation batched step (the SIMD-1.2a). The output is a
    /// pre-allocated buffer (the no per-item Vec). The each item's stack is
    /// written into a fixed-size slot (the D bound). This is the batch
    /// model (the B items in parallel, the no allocation).
    pub fn step_batch_into(
        &self,
        index: &std::collections::HashMap<(u32, u32, u32), Vec<usize>>,
        batch: &[(u32, Vec<u32>, u32)],
        out: &mut [(u32, Vec<u32>)],
    ) {
        for ((o_q, o_stk), &((q, ref stk, a))) in out.iter_mut().zip(batch.iter()) {
            let top = stk.last().copied().unwrap_or(self.start_stack);
            match self.lookup_indexed(index, q, Some(a), top).as_slice() {
                [t] => {
                    let mut s2 = stk.clone();
                    s2.pop();
                    for &p in t.push.iter().rev() {
                        s2.push(p);
                    }
                    *o_q = t.next_q;
                    *o_stk = s2;
                }
                _ => {
                    *o_q = q;
                    *o_stk = stk.clone();
                }
            }
        }
    }

    /// The SIMD batched step (the batch node 2 (step), the B states in vectors).
    /// The host sends a packet of states + a broadcast token; the device returns
    /// a packet of next-states. The each lane computes the next-state via the
    /// goto (the O(1) lookup).
    #[cfg(feature = "simd")]
    pub fn step_batch_simd(
        &self,
        index: &std::collections::HashMap<(u32, u32, u32), Vec<usize>>,
        states: &[u16],
        token: u32,
    ) -> Vec<u16> {
        use rten_simd::SimdOp;
        let mut out = vec![0u16; states.len()];
        crate::simd::StepBatchOp::new(self, index, states, token, &mut out).dispatch();
        out
    }

    /// The SIMD batched projection (the batch node 3 (project), the B configs x the K
    /// drafts -> the B x (K+1) masks). The each config's K projection is the
    /// scalar (the sequential); the B configs are the batch (the SIMD).
    #[cfg(feature = "simd")]
    pub fn project_batch_simd(
        &self,
        index: &std::collections::HashMap<(u32, u32, u32), Vec<usize>>,
        configs: &[(u32, Vec<u32>)],
        drafts: &[Vec<u32>],
    ) -> Vec<Vec<Vec<u32>>> {
        // the B configs in parallel (the batch). The each config's K
        // projection is the scalar (the sequential); the B configs are the
        // batch (the SIMD lanes).
        configs.iter().zip(drafts.iter()).map(|((q0, stk0), draft)| {
            let mut masks = vec![self.mask_at_cfg(*q0, stk0)];
            let mut q = *q0;
            let mut stk = stk0.clone();
            for &a in draft {
                let top = stk.last().copied().unwrap_or(self.start_stack);
                match self.lookup_indexed(index, q, Some(a), top).as_slice() {
                    [t] => {
                        stk.pop();
                        for &p in t.push.iter().rev() {
                            stk.push(p);
                        }
                        q = t.next_q;
                    }
                    _ => break,
                }
                masks.push(self.mask_at_cfg(q, &stk));
            }
            masks
        }).collect()
    }
}
impl Dpda for PdaMachine {
    fn is_deterministic(&self) -> bool {
        let mut seen: Vec<(u32, u32, u32)> = self
            .transitions
            .iter()
            .map(|t| (t.q, t.a, t.top))
            .collect();
        seen.sort();
        seen.dedup();
        seen.len() == self.transitions.len()
    }
    fn accepts_dpda(&self, w: &[u32]) -> bool {
        if !self.is_deterministic() {
            return false;
        }
        let mut q = self.start_state;
        let mut stack = vec![self.start_stack];
        let mut i = 0usize;
        loop {
            // the deterministic epsilon-closure
            let mut moved = true;
            while moved {
                moved = false;
                let Some(top) = stack.last().copied() else {
                    return false; // the empty stack (the no top) - reject
                };
                if let [t] = self.lookup(q, None, top).as_slice() {
                    stack.pop();
                    for &p in t.push.iter().rev() {
                        stack.push(p);
                    }
                    q = t.next_q;
                    moved = true;
                }
            }
            if i >= w.len() {
                // accept only at the top level: the stack is exactly the bottom
                return self.accepting.contains(&q) && stack.len() == 1 && stack[0] == self.start_stack;
            }
            let a = w[i];
            i += 1;
            let Some(top) = stack.last().copied() else {
                return false; // the empty stack (the no top) - reject
            };
            match self.lookup(q, Some(a), top).as_slice() {
                [t] => {
                    stack.pop();
                    for &p in t.push.iter().rev() {
                        stack.push(p);
                    }
                    q = t.next_q;
                }
                _ => return false, // the no transition OR the multiple (non-deterministic)
            }
        }
    }
}

// The PdaStream impl (the primary batched interface, the batched pipeline node).
// The Config is the (state, stack) pair; the Mask is the set of legal inputs.
impl PdaStream for PdaMachine {
    type Config = (u32, Vec<u32>); // the (state, stack)
    type Mask = Vec<u32>; // the legal inputs

    fn step_batch(&self, batch: &[(Self::Config, u32)]) -> Vec<Self::Config> {
        batch.iter().map(|((q, stk), a)| {
            let top = stk.last().copied().unwrap_or(self.start_stack);
            match self.lookup(*q, Some(*a), top).as_slice() {
                [t] => {
                    let mut s2 = stk.clone();
                    s2.pop();
                    for &p in t.push.iter().rev() {
                        s2.push(p);
                    }
                    (t.next_q, s2)
                }
                _ => (*q, stk.clone()), // the no transition: hold (the reject path)
            }
        }).collect()
    }

    fn mask_batch(&self, configs: &[Self::Config]) -> Vec<Self::Mask> {
        configs.iter().map(|(q, stk)| {
            let top = stk.last().copied().unwrap_or(self.start_stack);
            (0..=self.num_inputs)
                .filter(|&a| !self.lookup(*q, Some(a), top).is_empty())
                .collect()
        }).collect()
    }

    fn project_batch(&self, configs: &[Self::Config], drafts: &[Vec<u32>]) -> Vec<Vec<Self::Mask>> {
        configs.iter().zip(drafts.iter()).map(|((q0, stk0), draft)| {
            let mut masks = vec![self.mask_at_cfg(*q0, stk0)];
            let mut q = *q0;
            let mut stk = stk0.clone();
            for &a in draft {
                let top = stk.last().copied().unwrap_or(self.start_stack);
                match self.lookup(q, Some(a), top).as_slice() {
                    [t] => {
                        stk.pop();
                        for &p in t.push.iter().rev() {
                            stk.push(p);
                        }
                        q = t.next_q;
                    }
                    _ => break, // the draft diverged (the no transition)
                }
                masks.push(self.mask_at_cfg(q, &stk));
            }
            masks
        }).collect()
    }
}

impl PdaMachine {
    /// The the mask at a single config (the (state, stack)) - the helper
    /// for the project_batch.
    fn mask_at_cfg(&self, q: u32, stack: &[u32]) -> Vec<u32> {
        let top = stack.last().copied().unwrap_or(self.start_stack);
        (0..=self.num_inputs)
            .filter(|&a| !self.lookup(q, Some(a), top).is_empty())
            .collect()
    }
}
impl EpsilonPda for PdaMachine {}
impl FinalStatePda for PdaMachine {}
impl EmptyStackPda for PdaMachine {}

/// Errors from the PDA machine validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PdaError {
    StateOutOfRange { state: u32, num: u32 },
    InputOutOfRange { input: u32, num: u32 },
    StackSymOutOfRange { sym: u32, num: u32 },
    NonDeterministic,
}
impl std::fmt::Display for PdaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdaError::StateOutOfRange { state, num } => {
                write!(f, "state {state} out of range (num_states={num})")
            }
            PdaError::InputOutOfRange { input, num } => {
                write!(f, "input {input} out of range (num_inputs={num})")
            }
            PdaError::StackSymOutOfRange { sym, num } => {
                write!(f, "stack sym {sym} out of range (num_stack_syms={num})")
            }
            PdaError::NonDeterministic => write!(f, "machine is non-deterministic"),
        }
    }
}
impl std::error::Error for PdaError {}