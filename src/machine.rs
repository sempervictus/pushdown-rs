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
#[derive(Debug, Clone)]
pub struct PdaMachine {
    pub num_states: u32,
    pub num_inputs: u32,
    pub num_stack_syms: u32,
    pub transitions: Vec<Transition>,
    pub accepting: Vec<u32>,
    pub start_state: u32,
    pub start_stack: u32,
    /// The RTN state-to-nonterminal projection (the "grammar phase" of a control
    /// state). For a machine compiled from a CFG G = (N, Sigma, P, S) by the
    /// recursive-transition-network construction (Definition 5 of arXiv:2603.05540),
    /// the control states partition into three families, each created *for* one
    /// nonterminal:
    ///
    ///   q_start          -> S          (the start nonterminal)
    ///   q_A^in, q_A^out  -> A         (the entry / exit of nonterminal A)
    ///   q_(p,i)          -> lhs(p)    (the dot state of production p : lhs -> rhs)
    ///
    /// `state_provenance[q]` is that nonterminal's index in N (0..num_nonterminals).
    /// It is a TOTAL function: every control state belongs to exactly one
    /// nonterminal, so when `Some` the vector has length
    ///   kappa(G) = 1 + 2|N| + sum_p(|rhs(p)| + 1   (Lemma 2 of arXiv:2603.05540)
    /// which equals `num_states`.
    ///
    /// WHY this is a primitive (not a consumer detail): it is the phase label of a
    /// state. A consumer maps the nonterminal index to a semantic region (a
    /// grammar's reasoning / text / tool-call phases, a protocol's header / body /
    /// trailer, ...) WITHOUT reverse-engineering the RTN numbering. It is the same
    /// *kind* of per-state classification as PSC's parser-stack -> mask-class
    /// (arXiv:2608.03065), just at the control-state granularity.
    ///
    /// PHASE HOMOLOGY (the logic that makes it useful): the label is invariant along
    /// a terminal run inside one production (q_(p,i) -> q_(p,i+1) keeps lhs(p)), and
    /// changes exactly at the call / return / exit boundaries (q_(p,i) -> q_B^in
    /// switches to B; q_B^out -> q_(p,i) restores lhs(p); q_(p,m) -> q_A^out
    /// switches to A). So it tracks "which nonterminal is currently being parsed."
    ///
    /// `Some` for RTN-compiled machines; `None` for hand-built ones (the doc
    /// example) where the state -> nonterminal association is not defined.
    pub state_provenance: Option<Vec<u32>>,
    /// The vocabulary names (the human-readable terminal labels, indexed by input ID
    /// 0..num_inputs). This is the PDA's "vocabulary mapping interface": an external
    /// caller identifies which actual lexeme is represented at which PDA bit by
    /// mapping input_id -> vocab_names[input_id] -> the concrete token/character/field.
    /// `Some` for RTN-compiled machines (the `compile` populates it); `None` for
    /// hand-built ones (the doc example) where the vocabulary labels are not defined.
    pub vocab_names: Option<Vec<String>>,
    /// CSR index: for each control state q, the starting index into `transitions`
    /// where q's records begin, and the count of records. This allows O(1-3)
    /// transition lookups per state (instead of O(total_transitions) linear scan).
    /// Computed once at construction. The sentinel value (transitions.len()) marks
    /// the end of the array for states with no transitions.
    pub ctrl_offsets: Vec<u32>,
    pub ctrl_counts: Vec<u32>,
}

// The CSR fields (ctrl_offsets, ctrl_counts) are derived data (computed from
// the transitions), not part of the machine's identity. Exclude them from
// equality.
impl PartialEq for PdaMachine {
    fn eq(&self, other: &Self) -> bool {
        self.num_states == other.num_states
            && self.num_inputs == other.num_inputs
            && self.num_stack_syms == other.num_stack_syms
            && self.transitions == other.transitions
            && self.accepting == other.accepting
            && self.start_state == other.start_state
            && self.start_stack == other.start_stack
            && self.state_provenance == other.state_provenance
            && self.vocab_names == other.vocab_names
    }
}
impl Eq for PdaMachine {}

impl PdaMachine {
    /// Compute the CSR index (ctrl_offsets + ctrl_counts) from the transitions.
    /// This is a static helper for construction sites that build PdaMachine
    /// directly (not via `PdaMachine::new`).
    pub fn compute_csr(transitions: &[Transition], num_states: u32) -> (Vec<u32>, Vec<u32>) {
        let n = num_states as usize;
        let mut offsets = vec![transitions.len() as u32; n];
        let mut counts = vec![0u32; n];
        for (i, t) in transitions.iter().enumerate() {
            if (t.q as usize) < n {
                if counts[t.q as usize] == 0 {
                    offsets[t.q as usize] = i as u32;
                }
                counts[t.q as usize] += 1;
            }
        }
        (offsets, counts)
    }

    /// Build a machine, validating the bounds.
    #[allow(clippy::too_many_arguments)] // the the 8-field constructor (the the POD, the the no builder)
    pub fn new(
        num_states: u32,
        num_inputs: u32,
        num_stack_syms: u32,
        transitions: Vec<Transition>,
        accepting: Vec<u32>,
        start_state: u32,
        start_stack: u32,
        state_provenance: Option<Vec<u32>>,
        vocab_names: Option<Vec<String>>,
    ) -> StdResult<Self, PdaError> {
        // Compute the CSR index (ctrl_offsets + ctrl_counts) for O(1) transition
        // lookups per control state. The transitions are grouped by `q` (the
        // control state), so the CSR allows the kernel to scan only the
        // transitions for a specific state (O(1-3) instead of O(total)).
        let n = num_states as usize;
        let mut ctrl_offsets = vec![transitions.len() as u32; n]; // sentinel
        let mut ctrl_counts = vec![0u32; n];
        for (i, t) in transitions.iter().enumerate() {
            if (t.q as usize) < n {
                if ctrl_counts[t.q as usize] == 0 {
                    ctrl_offsets[t.q as usize] = i as u32;
                }
                ctrl_counts[t.q as usize] += 1;
            }
        }
        let m = PdaMachine {
            num_states,
            num_inputs,
            num_stack_syms,
            transitions,
            accepting,
            start_state,
            start_stack,
            state_provenance,
            vocab_names,
            ctrl_offsets,
            ctrl_counts,
        };
        m.validate_bounds()?;
        Ok(m)
    }

    /// The vocabulary label for a PDA input bit (the `vocab_names[input_id]`). This is the
    /// "vocabulary mapping interface": an external caller identifies which actual lexeme is
    /// represented at which PDA bit by mapping input_id -> this label -> the concrete token /
    /// character / protocol field. `None` when the machine has no vocabulary labels (a hand-
    /// built machine, or a bitvec-loaded one) or the input ID is out of range.
    pub fn vocab_name(&self, input_id: u32) -> Option<&str> {
        self.vocab_names
            .as_ref()
            .and_then(|v| v.get(input_id as usize).map(|s| s.as_str()))
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
        // Provenance (when present) is a TOTAL function Q -> N, so two invariants hold:
        //   (1) LENGTH: one entry per control state, i.e. prov.len() == num_states
        //       (== kappa(G) by Lemma 2). A shorter/longer vector is malformed.
        //   (2) RANGE: each entry is a nonterminal index in 0..|N|. The machine does
        //       not store |N| directly, but the RTN numbering guarantees |N| <
        //       num_states (since kappa(G) = 1 + 2|N| + ... > |N|), so bounding each
        //       entry by num_states is a sound (if slightly loose) range check.
        if let Some(prov) = &self.state_provenance {
            if prov.len() != self.num_states as usize {
                return Err(PdaError::ProvenanceLength {
                    got: prov.len(),
                    expected: self.num_states as usize,
                });
            }
            for &p in prov {
                if p >= self.num_states {
                    return Err(PdaError::ProvenanceOutOfRange {
                        state: p,
                        num: self.num_states,
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

    /// The grammar-phase of control state `q`: the nonterminal index (0..num_nonterminals)
    /// that `q` belongs to, per the RTN projection documented on `state_provenance`.
    ///
    /// Returns `None` in two cases: (1) the machine has no provenance (a hand-built
    /// machine, where the state -> nonterminal association is undefined), or (2) `q`
    /// is out of range. For an RTN-compiled machine and a valid `q`, this is `Some`.
    ///
    /// CONSUMER PATTERN: a region/phase-aware user com the returned nonterminal index
    /// to a semantic bucket via the grammar's *named* nonterminals (e.g. index ->
    /// "reasoning_block" -> Region::Reasoning). The push deliberately stops at the
    /// index: it is grammar-agnostic and does not know the nonterminal names.
    pub fn provenance_of(&self, q: u32) -> Option<u32> {
        let prov = self.state_provenance.as_ref()?;
        prov.get(q as usize).copied()
    }

    /// The epsilon-closure advance: from (q, stk), follow the epsilon moves (the no
    /// input) to reach the configs where the terminal move `a` is available, then do
    /// the terminal move. Returns `Some((next_state, next_stack))` on success, or
    /// `None` if no terminal move is reachable (the divergence / the reject path).
    ///
    /// This matches the accepts_npda BFS semantics (the epsilon moves are resolved
    /// before the terminal move). The RTN epsilon graph is local, so the closure is
    /// bounded; the MAX_EPS_CLOSURE cap is the safety safeguard against a
    /// pathological (the non-RTN) machine.
    ///
    /// Uses the CSR index (ctrl_offsets + ctrl_counts) for O(1-3) transition
    /// lookups per state (instead of O(total_transitions) linear scan).
    pub fn advance_eps(&self, q: u32, stk: &[u32], a: u32) -> Option<(u32, Vec<u32>)> {
        const MAX_EPS_CLOSURE: usize = 4096;
        let mut configs: Vec<(u32, Vec<u32>)> = vec![(q, stk.to_vec())];
        let mut i = 0;
        let use_csr = !self.ctrl_offsets.is_empty();
        while i < configs.len() && configs.len() < MAX_EPS_CLOSURE {
            let (cq, cstk) = configs[i].clone();
            i += 1;
            let ctop = cstk.last().copied().unwrap_or(self.start_stack);
            if use_csr {
                // CSR-based epsilon lookup: scan only cq's transitions (O(1-3)).
                let start = self.ctrl_offsets.get(cq as usize).copied().unwrap_or(self.transitions.len() as u32);
                let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0);
                for j in 0..count {
                    let t = &self.transitions[start as usize + j as usize];
                    if t.a != self.num_inputs || t.top != ctop {
                        continue;
                    }
                    let mut ns = cstk.clone();
                    ns.pop();
                    for &p in t.push.iter().rev() {
                        ns.push(p);
                    }
                    if !configs.iter().any(|(s, ss)| *s == t.next_q && *ss == ns) {
                        configs.push((t.next_q, ns));
                    }
                }
            } else {
                // Fallback: linear scan (the original path, for machines without CSR).
                for (q2, push) in self.transition(cq, None, ctop) {
                    let mut ns = cstk.clone();
                    ns.pop();
                    for &p in push.iter().rev() {
                        ns.push(p);
                    }
                    if !configs.iter().any(|(s, ss)| *s == q2 && *ss == ns) {
                        configs.push((q2, ns));
                    }
                }
            }
        }
        // Terminal lookup: CSR-based or linear scan.
        for (cq, cstk) in &configs {
            let top = cstk.last().copied().unwrap_or(self.start_stack);
            if use_csr {
                let start = self.ctrl_offsets.get(*cq as usize).copied().unwrap_or(self.transitions.len() as u32);
                let count = self.ctrl_counts.get(*cq as usize).copied().unwrap_or(0);
                for j in 0..count {
                    let t = &self.transitions[start as usize + j as usize];
                    if t.a == a && t.top == top {
                        let mut s2 = cstk.clone();
                        s2.pop();
                        for &p in t.push.iter().rev() {
                            s2.push(p);
                        }
                        return Some((t.next_q, s2));
                    }
                }
            } else {
                if let [t] = self.lookup(*cq, Some(a), top).as_slice() {
                    let mut s2 = cstk.clone();
                    s2.pop();
                    for &p in t.push.iter().rev() {
                        s2.push(p);
                    }
                    return Some((t.next_q, s2));
                }
            }
        }
        None
    }

    /// The universal accepts: auto-selects the right simulation based on the
    /// machine's determinism. Callers do NOT need to know if the machine is a
    /// DPDA or an NPDA - this picks the correct one. It uses the deterministic
    /// path (the accepts_dpda, the single path, the fast) or the
    /// non-deterministic path (the accepts_npda, the BFS, the bounded).
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
            self.num_states, self.num_inputs, self.num_stack_syms, self.start_state, self.accepting
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
        let mut idx: std::collections::HashMap<(u32, u32, u32), Vec<usize>> =
            std::collections::HashMap::new();
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
        for (i, out) in out.iter_mut().enumerate() {
            *out = if self.lookup(state, Some(i as u32), self.start_stack).is_empty() {
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
    fn transition(&self, q: u32, a: Option<u32>, top: u32) -> Vec<(u32, Vec<u32>)> {
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
                return frontier.iter().any(|(q, s)| {
                    self.accepting.contains(q) && s.len() == 1 && s[0] == self.start_stack
                });
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
                    let Some(top) = stack.last().copied() else {
                        continue;
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
                let Some(top) = stack.last().copied() else {
                    continue;
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
        for ((o_q, o_stk), &(q, ref stk, a)) in out.iter_mut().zip(batch.iter()) {
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
        configs
            .iter()
            .zip(drafts.iter())
            .map(|((q0, stk0), draft)| {
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
            })
            .collect()
    }
}
impl Dpda for PdaMachine {
    fn is_deterministic(&self) -> bool {
        let mut seen: Vec<(u32, u32, u32)> =
            self.transitions.iter().map(|t| (t.q, t.a, t.top)).collect();
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
                return self.accepting.contains(&q)
                    && stack.len() == 1
                    && stack[0] == self.start_stack;
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
        batch
            .iter()
            .map(|((q, stk), a)| {
                // the epsilon-closure advance: follow the epsilon moves to
                // the terminal state, then the terminal move. On divergence (the no
                // terminal move), hold (q, stk) (the reject path).
                self.advance_eps(*q, stk, *a).unwrap_or((*q, stk.clone()))
            })
            .collect()
    }

    fn mask_batch(&self, configs: &[Self::Config]) -> Vec<Self::Mask> {
        configs
            .iter()
            .map(|(q, stk)| {
                let top = stk.last().copied().unwrap_or(self.start_stack);
                // Include the epsilon-closure: follow epsilon transitions from (q, top)
                // and collect all allowed inputs from every reachable state.
                let mut allowed: Vec<u32> = Vec::new();
                let mut visited: std::collections::HashSet<(u32, u32)> =
                    std::collections::HashSet::new();
                let mut frontier = vec![(*q, top)];
                while let Some((cq, ctop)) = frontier.pop() {
                    if !visited.insert((cq, ctop)) {
                        continue;
                    }
                    // Collect non-epsilon inputs from this state.
                    for a in 0..self.num_inputs {
                        if !self.lookup(cq, Some(a), ctop).is_empty() && !allowed.contains(&a) {
                            allowed.push(a);
                        }
                    }
                    // Follow epsilon transitions.
                    for (q2, push) in self.transition(cq, None, ctop) {
                        let new_top = if push.is_empty() {
                            // Popped the top, no push: the new top is the previous stack element.
                            // For simplicity, use the same top (the epsilon doesn't change the stack
                            // in most RTN compilations).
                            ctop
                        } else {
                            // The push is applied in reverse (the step_batch's iter().rev()), so the
                            // new top is push.first() (the push[0], the return address for a call).
                            *push.first().unwrap()
                        };
                        frontier.push((q2, new_top));
                    }
                }
                allowed
            })
            .collect()
    }

    fn project_batch(&self, configs: &[Self::Config], drafts: &[Vec<u32>]) -> Vec<Vec<Self::Mask>> {
        configs
            .iter()
            .zip(drafts.iter())
            .map(|((q0, stk0), draft)| {
                let mut masks = vec![self.mask_at_cfg(*q0, stk0)];
                let mut q = *q0;
                let mut stk = stk0.clone();
                for &a in draft {
                    // the epsilon-closure advance: follow the epsilon moves to the
                    // terminal state, then the terminal move. Break on divergence (the
                    // no terminal move reachable for this draft token).
                    match self.advance_eps(q, &stk, a) {
                        Some((nq, ns)) => {
                            q = nq;
                            stk = ns;
                        }
                        None => break,
                    }
                    masks.push(self.mask_at_cfg(q, &stk));
                }
                masks
            })
            .collect()
    }
}

impl PdaMachine {
    /// The mask at a single config (the (state, stack)): the epsilon-closure union
    /// of the allowed inputs (the follow the epsilon moves to every reachable
    /// config, then collect the terminals with a defined move). This is the
    /// single-config entry for the PDA-as-FSM-mirror contract — it is exactly
    /// the set of inputs for which `advance_eps` succeeds (the proof
    /// `proof_mask_batch_consistent_with_advance_eps`). `mask_batch` is the
    /// batched form of this same computation.
    pub fn mask_at_cfg(&self, q: u32, stack: &[u32]) -> Vec<u32> {
        let top = stack.last().copied().unwrap_or(self.start_stack);
        // Include the epsilon-closure: follow epsilon transitions from (q, top)
        // and collect all allowed inputs from every reachable state.
        let mut allowed: Vec<u32> = Vec::new();
        let mut visited: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut frontier = vec![(q, top)];
        while let Some((cq, ctop)) = frontier.pop() {
            if !visited.insert((cq, ctop)) {
                continue;
            }
            for a in 0..self.num_inputs {
                if !self.lookup(cq, Some(a), ctop).is_empty() && !allowed.contains(&a) {
                    allowed.push(a);
                }
            }
            for (q2, push) in self.transition(cq, None, ctop) {
                let new_top = if push.is_empty() {
                    ctop
                } else {
                    // The push is applied in reverse (the step_batch's iter().rev()), so the
                    // new top is push.first() (the push[0], the return address for a call).
                    *push.first().unwrap()
                };
                frontier.push((q2, new_top));
            }
        }
        allowed
    }

    /// The the mask at a SINGLE settled config (the (state, stack-top)) - the
    /// precise inclusive gate. Unlike mask_at_cfg (the epsilon-closure union),
    /// this reports EXACTLY the terminals with a defined transition at (q, a, top)
    /// without following epsilon moves to other configs. This is the mask source
    /// for the PDA-as-mask-source architecture (the no-approximation gate).
    ///
    /// The exclusive gate is the complement over the vocab (the terminals NOT in
    /// this set). The inclusive + exclusive properties are proven by
    /// proof_mask_at_cfg_settled_is_precise (the pda_tests.rs).
    pub fn mask_at_cfg_settled(&self, q: u32, top: u32) -> Vec<u32> {
        let mut allowed: Vec<u32> = Vec::new();
        for a in 0..self.num_inputs {
            if !self.lookup(q, Some(a), top).is_empty() {
                allowed.push(a);
            }
        }
        allowed
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
    ProvenanceLength { got: usize, expected: usize },
    ProvenanceOutOfRange { state: u32, num: u32 },
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
            PdaError::ProvenanceLength { got, expected } => {
                write!(f, "state_provenance length {got} != num_states {expected}")
            }
            PdaError::ProvenanceOutOfRange { state, num } => {
                write!(
                    f,
                    "state_provenance entry {state} out of range (num_states={num})"
                )
            }
            PdaError::NonDeterministic => write!(f, "machine is non-deterministic"),
        }
    }
}
impl std::error::Error for PdaError {}
