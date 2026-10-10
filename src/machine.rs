//! The concrete PDA machine (the 7-tuple with u32 IDs) + the variant trait
//! impls (the NPDA, the DPDA, the epsilon, the final-state, the empty-stack).

use crate::pda::{DisplacementPda, Dpda, EmptyStackPda, EpsilonPda, FinalStatePda, Npda, Pda, PdaStream};
use crate::compile::Cfg;
use std::collections::HashSet;
use std::result::Result as StdResult;

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
    /// The SoA flat arrays (the SIMD-gatherable layout): the transitions' fields
    /// are flattened into parallel arrays (the a, the top, the next_q). This is
    /// the layout the AVX-512 VGATHER (the CSR gather) reads: the B lanes
    /// gather their flat_a / flat_top values in parallel (the no per-lane scalar).
    /// Derived data (the computed from the transitions), excluded from equality.
    pub flat_a: Vec<u32>,
    pub flat_top: Vec<u32>,
    pub flat_next_q: Vec<u32>,
    /// The precomputed epsilon-closure table (the owl owl `epsilon_closure_for_state`
    /// pattern, the the automaton-epsilon-closure.c). For each (state, top) pair, the
    /// set of (state, top) pairs reachable via epsilon moves. This is computed ONCE at
    /// construction (the the automcompute_closure_table`), and looked up in O(1) per
    /// `mask_at_cfg` call (the the no the per-call BFS). The owl prior art: the
    /// `automaton_compute_epsilon_closure` precomputes the closure per state, and the
    /// `follow_transition` step is O(1) (the the no the per-step closure recomputation).
    ///
    /// The table is keyed by (state, top) — the the `mask_at_cfg`'s input. The value is
    /// the set of (state, top) pairs in the epsilon closure. For the VPA / the bounded
    /// case, the closure is small (the the O(num_states x num_stack_syms) pairs). For the
    /// star-loop case, the closure is larger (the the growth cycle), but still bounded by
    /// the num_states x num_stack_syms domain.
    ///
    /// Derived data (the computed from the transitions), excluded from equality (the the
    /// same as the CSR fields). Computed eagerly at construction (the the `new` + the
    /// `compile` + the `from_bitvec`), looked up in O(1) per mask_at_cfg call.
    pub closure_table: Vec<Vec<(u32, u32)>>,
}

// The CSR fields (ctrl_offsets, ctrl_counts) are derived data (computed from
// the transitions), not part of the machine's identity. Exclude them from
// equality. vocab_names is likewise excluded: it is a CPU-side concern that the
// bitvec POD deliberately does not carry (the consumer re-derives it from the
// grammar), so a to_bitvec -> from_bitvec round-trip (Some vs None) must still
// compare equal.
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
    }
}
impl Eq for PdaMachine {}

/// The PDA kind, determined programmatically from the machine's structure. This is the
/// dispatch key: the caller inspects the kind and uses the tight algorithm + bound for it,
/// instead of the loose general-NPDA path for every machine.
///
/// The kinds (the the trait hierarchy in pda.rs, the the programmatic determination):
/// - `Deterministic`: no `(q, a, top)` has two transitions (the DPDA, the single path). The
///   epsilon closure is a single path (the no the frontier BFS). The tight depth bound is
///   `max_stack_depth` when the machine has no self-recursive epsilon growth cycle (the
///   bounded-nesting case), else `num_states` (the star-loop case).
/// - `VisiblyPushdown`: every input-consuming move's stack-op class (the call / the return /
///   the internal) is a function of the INPUT SYMBOL alone (the no the state). This is the
///   VPL case (the JSON well-nested, the the xml, the the eBPF call/return). The stack depth
///   is bounded by the nesting depth (the max_stack_depth, the tight). The mask is O(1) (the
///   settled gate, the no the closure BFS, because the VPA has no epsilon moves that change
///   the stack top in a state-dependent way).
/// - `NonDeterministic`: the general NPDA (the choices, the one_or_more loops, the ambiguous
///   grammars). The frontier BFS is required (the advance_eps_set, the full-stack dedup). The
///   depth bound is `num_states` (the pigeonhole, the the star-loop re-entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdaKind {
    Deterministic,
    VisiblyPushdown,
    NonDeterministic,
}

impl PdaMachine {
    /// Compute the SoA flat arrays (the flat_a, the flat_top, the flat_next_q)
    /// from the transitions (the SIMD-gatherable layout). This is a static
    /// helper for construction sites that build PdaMachine directly.
    pub fn compute_flat_arrays(transitions: &[Transition]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut flat_a = Vec::with_capacity(transitions.len());
        let mut flat_top = Vec::with_capacity(transitions.len());
        let mut flat_next_q = Vec::with_capacity(transitions.len());
        for t in transitions {
            flat_a.push(t.a);
            flat_top.push(t.top);
            flat_next_q.push(t.next_q);
        }
        (flat_a, flat_top, flat_next_q)
    }

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

    /// Compute the precomputed epsilon-closure table (the the owl
    /// `automaton_compute_epsilon_closure` pattern, the the
    /// automaton-epsilon-closure.c). For each (state, top) pair, the table stores
    /// the set of (state, top) pairs reachable via epsilon moves. This is computed
    /// ONCE at construction, and looked up in O(1) per `mask_at_cfg` call (the the
    /// no the per-call BFS).
    ///
    /// The owl prior art: the `automaton_compute_epsilon_closure` precomputes the
    /// closure per state (the the `epsilon_closure_for_state[i]`), and the
    /// `follow_transition` step is O(1) (the the no the per-step closure
    /// recomputation). My transcription: the closure is per (state, top) pair (the the
    /// PDA's config space), and the `mask_at_cfg` looks up the closure in O(1).
    ///
    /// The closure is the least fixed point of the epsilon relation (the the
    /// Bouajjani 1997 summar). It terminates when no new (state, top) pair is
    /// reachable (the the finite domain, the num_states x num_stack_syms).
    pub fn compute_closure_table(&mut self) {
        let n_states = self.num_states as usize;
        let n_tops = self.num_stack_syms as usize;
        // The table is indexed by (state, top) -> the set of (state, top) pairs in the closure.
        // The size is num_states x num_stack_syms (the the finite domain).
        let mut table: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_states * n_tops];
        for q in 0..n_states {
            for top in 0..n_tops {
                let idx = q * n_tops + top;
                // The epsilon closure of (q, top): the BFS over the (state, top) pairs.
                let mut closed: Vec<(u32, u32)> = vec![(q as u32, top as u32)];
                let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
                seen.insert((q as u32, top as u32));
                let mut i = 0;
                while i < closed.len() {
                    let (cq, ctop) = closed[i];
                    i += 1;
                    // The epsilon moves from (cq, ctop) (the the a == num_inputs, the the top == ctop).
                    if !self.ctrl_offsets.is_empty() {
                        let start = self.ctrl_offsets.get(cq as usize).copied().unwrap_or(self.transitions.len() as u32) as usize;
                        let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0) as usize;
                        for j in 0..count {
                            let t = &self.transitions[start + j];
                            if t.a != self.num_inputs || t.top != ctop {
                                continue;
                            }
                            // The new top (the the push replaces the top, the the new top is the push[0] or the ctop if the push is empty).
                            let new_top = if t.push.is_empty() {
                                ctop
                            } else {
                                *t.push.first().unwrap()
                            };
                            if seen.insert((t.next_q, new_top)) {
                                closed.push((t.next_q, new_top));
                            }
                        }
                    } else {
                        for (q2, push) in self.transition(cq, None, ctop) {
                            let new_top = if push.is_empty() {
                                ctop
                            } else {
                                *push.first().unwrap()
                            };
                            if seen.insert((q2, new_top)) {
                                closed.push((q2, new_top));
                            }
                        }
                    }
                }
                table[idx] = closed;
            }
        }
        self.closure_table = table;
    }

    /// The precomputed epsilon-closure lookup (the the O(1) replacement for the per-call BFS).
    /// Returns the set of (state, top) pairs in the epsilon closure of (q, top).
    pub fn closure_at(&self, q: u32, top: u32) -> &[(u32, u32)] {
        let n_tops = self.num_stack_syms as usize;
        let idx = (q as usize) * n_tops + (top as usize);
        &self.closure_table[idx]
    }

    /// Build a machine, validating the bounds.
    #[allow(clippy::too_many_arguments)] // the the 8-field constructor (the the POD, the the no builder)
    pub fn new(
        num_states: u32,
        num_inputs: u32,
        num_stack_syms: u32,
        mut transitions: Vec<Transition>,
        accepting: Vec<u32>,
        start_state: u32,
        start_stack: u32,
        state_provenance: Option<Vec<u32>>,
        vocab_names: Option<Vec<String>>,
    ) -> StdResult<Self, PdaError> {
        // Sort the transitions by (q, a, top) so the CSR computed below is
        // VALID: each control state's transitions must be contiguous. The CSR
        // (the first-occurrence + count) assumes per-state contiguity, which
        // the caller's array order does not guarantee (the RTN build pushes in
        // production order). The sort is a reordering (the transition set is
        // unchanged), so it preserves the machine's language + the identity
        // chain.
        transitions.sort_by_key(|t| (t.q, t.a, t.top));
        // Compute the CSR index (ctrl_offsets + ctrl_counts) for O(1) transition
        // lookups per control state. The transitions are grouped by `q` (the
        // control state, the sorted above), so the CSR allows the kernel to scan
        // only the transitions for a specific state (O(1-3) instead of O(total)).
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
        let (flat_a, flat_top, flat_next_q) = Self::compute_flat_arrays(&transitions);
        let mut m = PdaMachine {
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
            flat_a,
            flat_top,
            flat_next_q,
            closure_table: Vec::new(), // the filled by the compute_closure_table below
        };
        m.validate_bounds()?;
        // The precomputed epsilon-closure table (the the owl automaton_compute_epsilon_closure
        // pattern). Computed once at construction, looked up in O(1) per mask_at_cfg call.
        m.compute_closure_table();
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
    /// before the terminal move). The closure is over the (state, stack-top) pairs,
    /// a finite domain of size num_states * num_stack_syms, so the BFS terminates
    /// when no new pair is reachable (the no the synthetic cap).
    ///
    /// Uses the CSR index (ctrl_offsets + ctrl_counts) for O(1-3) transition
    /// lookups per state (instead of O(total_transitions) linear scan).
    pub fn advance_eps(&self, q: u32, stk: &[u32], a: u32) -> Option<(u32, Vec<u32>)> {
        let use_csr = !self.ctrl_offsets.is_empty();
        // The top of the stack (the the stk.last()). An empty stack is coerced to
        // the start_stack (the bottom marker) - the PDA invariant (the the stack
        // always has the bottom, the an empty stack means "at the bottom"). This
        // coercion is the documented PDA semantics (the no a silent gap): the
        // cycle_dpda tests rely on the empty stack matching the bottom-marker
        // transitions.
        let start_top = stk.last().copied().unwrap_or(self.start_stack);
        // The epsilon closure, dispatched on the growth cycle (the consistent with the mask_at_cfg).
        // the full-stack (state, Vec) dedup when the machine has an epsilon growth cycle (the
        // star-loop re-entry, the the (state, top) dedup prematurely terminates the BFS), else
        // the (state, top) dedup (the VPA / the bounded case, the GreatGramma Prop 3.5 sound).
        let full_stack = self.has_epsilon_growth_cycle();
        let depth_bound = self.closure_depth_bound();
        let mut configs: Vec<(u32, Vec<u32>)> = vec![(q, stk.to_vec())];
        let mut i = 0;
        if full_stack {
            let mut seen: HashSet<(u32, Vec<u32>)> = HashSet::new();
            seen.insert((q, stk.to_vec()));
            while i < configs.len() {
                let (cq, cstk) = configs[i].clone();
                i += 1;
                if cstk.len() > depth_bound {
                    continue;
                }
                let ctop = cstk.last().copied().unwrap_or(self.start_stack);
                if use_csr {
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
                        if seen.insert((t.next_q, ns.clone())) {
                            configs.push((t.next_q, ns));
                        }
                    }
                } else {
                    for (q2, push) in self.transition(cq, None, ctop) {
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in push.iter().rev() {
                            ns.push(p);
                        }
                        if seen.insert((q2, ns.clone())) {
                            configs.push((q2, ns));
                        }
                    }
                }
            }
        } else {
            let mut seen: HashSet<(u32, u32)> = HashSet::new();
            seen.insert((q, start_top));
            while i < configs.len() {
                let (cq, cstk) = configs[i].clone();
                i += 1;
                if cstk.len() > depth_bound {
                    continue;
                }
                let ctop = cstk.last().copied().unwrap_or(self.start_stack);
                if use_csr {
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
                        let ns_top = ns.last().copied().unwrap_or(self.start_stack);
                        if seen.insert((t.next_q, ns_top)) {
                            configs.push((t.next_q, ns));
                        }
                    }
                } else {
                    for (q2, push) in self.transition(cq, None, ctop) {
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in push.iter().rev() {
                            ns.push(p);
                        }
                        let ns_top = ns.last().copied().unwrap_or(self.start_stack);
                        if seen.insert((q2, ns_top)) {
                            configs.push((q2, ns));
                        }
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

    /// The six-property #5 bounded-pushdown depth D: the maximum number of stack symbols any
    /// reachable stack holds. For an RTN-compiled PDA this is the max production length (the
    /// call pushes one return address, the nesting depth is bounded by the longest production).
    /// Concretely it is the max over all transitions of `push.len()` (a move replaces the top
    /// with `push`, so the stack can grow by at most `push.len() - 1` per move, and the deepest
    /// reachable stack is bounded by the max push length). This is the machine-derived bound that
    /// terminates the full-stack epsilon-closure BFS (the no a synthetic cap).
    pub fn max_stack_depth(&self) -> usize {
        self.transitions.iter().map(|t| t.push.len()).max().unwrap_or(1)
    }

    /// Determine the PDA kind from the machine's structure (the programmatic dispatch key).
    /// The order of checks (the most specific first,
    /// 1. VisiblyPushdown: the stack-op class is a function of the input symbol alone.
    /// 2. Deterministic: no (q, a, top) has two transitions.
    /// 3. NonDeterministic: the general case (the fallback).
    pub fn classify(&self) -> PdaKind {
        if self.is_visibly_pushdown() {
            return PdaKind::VisiblyPushdown;
        }
        if self.is_deterministic() {
            return PdaKind::Deterministic;
        }
        PdaKind::NonDeterministic
    }

    /// The VPA property: every input-consuming move's stack-op class (the call / the return /
    /// the internal) is determined by the INPUT SYMBOL `a` alone (the no the control state `q`).
    /// Concretely: for each input `a`, all transitions consuming `a` have the SAME push-length
    /// class (the call = push.len() >= 2, the return = push.len() == 0, the internal =
    /// push.len() == 1). If any input `a` is consumed with two different push-length classes
    /// (the state-dependent stack op), the machine is NOT visibly-pushdown.
    ///
    /// This is the property that makes the JSON / the XML / the eBPF grammars VPLs (the the
    /// well-nested, the the stack depth = the nesting depth, the bounded). The VPA's mask is O(1)
    /// (the settled gate, the no the closure BFS) because the stack top is fully determined by the
    /// input history (the the call/return matching, the no the state-dependent branching).
    fn is_visibly_pushdown(&self) -> bool {
        // The push-length class per input symbol: the call (>= 2), the return (== 0), the internal
        // (== 1). For the VPA,, each input `a` must have a SINGLE class across all states.
        let mut class_per_input: std::collections::HashMap<u32, u8> = std::collections::HashMap::new();
        for t in &self.transitions {
            if t.a == self.num_inputs {
                continue; // the epsilon move (the no input, the VPA class is over the input symbols)
            }
            let class: u8 = if t.push.len() >= 2 {
                0 // the call
            } else if t.push.is_empty() {
                1 // the return
            } else {
                2 // the internal (the push.len() == 1)
            };
            match class_per_input.get(&t.a) {
                Some(&prev) if prev != class => return false, // the state-dependent stack op (the no VPA)
                _ => {
                    class_per_input.insert(t.a, class);
                }
            }
        }
        true
    }

    /// The tight epsilon-closure depth bound for the machine's kind (the the dispatch). This is
    /// what terminates the full-stack BFS (the `mask_at_cfg` / the `advance_eps_set`) without a
    /// synthetic cap, AND without the loose `num_states` bound when the machine is actually
    /// bounded (the VPA / the deterministic no-growth-cycle case).
    ///
    /// - `VisiblyPushdown`: `max_stack_depth` (the nesting depth, the bounded, the tight).
    /// - `Deterministic`: `max_stack_depth` when there is no self-recursive epsilon growth cycle
    ///   (the bounded case), else `num_states` (the star-loop case, the pigeonhole).
    /// - `NonDeterministic`: `num_states` (the pigeonhole, the the star-loop re-entry grows the
    ///   stack by one return-address per iteration, the no bound tighter than the control states).
    pub fn closure_depth_bound(&self) -> usize {
        // The bound for the FRONTIER (the the advance_eps_set, the the loop preservation). The
        // VPA/D the Deterministic WITH a growth cycle (the the star-loop, the the one_or_more
        // re-entry, the the Qwen tool-call) needs the FULL depth (the num_states, the the
        // pigeonhole) to preserve the loop back-edge (the the stack grows by one return-address
        // per iteration, the the no the tight bound_stack_depth which truncates the loop). The
        // VPA / the Deterministic WITHOUT a growth cycle (the the bounded nesting, the the no
        // star-loop) the tight max_stack_depth (the the nesting depth). The NonDeterministic
        // (the the general NPDA, the the no VPA structure) is always the full num_states (the the
        // frontier, the the no the tight bound is not sound).
        match self.classify() {
            PdaKind::VisiblyPushdown => {
                if self.has_epsilon_growth_cycle() {
                    self.num_states as usize
                } else {
                    self.max_stack_depth()
                }
            }
            PdaKind::Deterministic => {
                if self.has_epsilon_growth_cycle() {
                    self.num_states as usize
                } else {
                    self.max_stack_depth()
                }
            }
            PdaKind::NonDeterministic => self.num_states as usize,
        }
    }

    /// Whether the machine's epsilon graph has a cycle with net stack growth (the star-loop
    /// re-entry: a state that pushes a return address and can reach itself, growing the stack by
    /// one per iteration). This is the six-property #5 unbounded case (the one_or_more loop). For
    /// a visibly-pushdown machine (the JSON well-nested, the no unbounded star), this is false
    /// (the stack is bounded by the nesting depth, the max production length).
    ///
    /// The detection: the epsilon graph (the states = nodes nodes, the epsilon moves = the edges)
    /// has a growth cycle iff it has an SCC of size > 1 (the cycle the self-loop) that contains a
    /// growth edge (the push.len() >= 2, the call that pushes a return address). For the RTN
    /// star-loop, the call/return re-entry forms such an SCC. For the bounded-nPA, the epsilon
    /// graph is acyclic (the the dot moves are input-driven, the no the epsilon cycles).
    pub fn has_epsilon_growth_cycle(&self) -> bool {
        let n = self.num_states as usize;
        // The epsilon adjacency (the state -> the successor states via the epsilon moves).
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        // The growth edges (the epsilon moves that push >= 2 symbols, the the call).
        let mut growth: Vec<(u32, u32)> = Vec::new();
        for t in &self.transitions {
            if t.a == self.num_inputs {
                if (t.q as usize) < n && (t.next_q as usize) < n {
                    adj[t.q as usize].push(t.next_q);
                    if t.push.len() >= 2 {
                        growth.push((t.q, t.next_q));
                    }
                }
            }
        }
        // The SCCar SCC (the Tarjan). A growth cycle exists iff some SCC of size > 1 contains a
        // growth edge (the the call that re-enters the same strongly-connected component).
        let mut index = vec![usize::MAX; n];
        let mut lowlink = vec![0usize; n];
        let mut on_stack = vec![false; n];
        let mut stack: Vec<u32> = Vec::new();
        let mut counter = 0usize;
        let mut scc_has_growth = vec![false; n]; // the per-SCC flag (the the SCC id -> the growth edge
        let mut scc_count = 0usize;

        fn strongconnect(
            v: u32,
            adj: &Vec<Vec<u32>>,
            growth: &Vec<(u32, u32)>,
            index: &mut Vec<usize>,
            lowlink: &mut Vec<usize>,
            on_stack: &mut Vec<bool>,
            stack: &mut Vec<u32>,
            counter: &mut usize,
            scc_has_growth: &mut Vec<bool>,
            scc_count: &mut usize,
        ) {
            let _n = adj.len();
            index[v as usize] = *counter;
            lowlink[v as usize] = *counter;
            *counter += 1;
            stack.push(v);
            on_stack[v as usize] = true;
            for &w in &adj[v as usize] {
                if index[w as usize] == usize::MAX {
                    strongconnect(
                        w, adj, growth, index, lowlink, on_stack, stack, counter, scc_has_growth, scc_count,
                    );
                    lowlink[v as usize] = lowlink[v as usize].min(lowlink[w as usize]);
                } else if on_stack[w as usize] {
                    lowlink[v as usize] = lowlink[v as usize].min(index[w as usize]);
                }
            }
            if lowlink[v as usize] == index[v as usize] {
                // The SCCar SCC. Pop it.
                let mut scc_size = 0usize;
                let mut scc_nodes: Vec<u32> = Vec::new();
                loop {
                    let w = stack.pop().unwrap();
                    on_stack[w as usize] = false;
                    scc_size += 1;
                    scc_nodes.push(w);
                    if w == v {
                        break;
                    }
                }
                // The growth cycle: the SCC has > 1 state (the the cycle). Any the RTN star-loop,
                // the call/return re-entry forms an SCC of size > 1. The stack grows by one
                // return address per iteration (the the push.len() can be 1 or 2), so ANY cycle
                // in the epsilon graph is a growth cycle (the the full-stack dedup is needed).
                if scc_size > 1 {
                    scc_has_growth[*scc_count] = true;
                }
                *scc_count += 1;
            }
        }

        for v in 0..n as u32 {
            if index[v as usize] == usize::MAX {
                strongconnect(
                    v, &adj, &growth, &mut index, &mut lowlink, &mut on_stack, &mut stack,
                    &mut counter, &mut scc_has_growth, &mut scc_count,
                );
            }
        }
        scc_has_growth.iter().any(|&g| g)
    }

    /// The NPDA epsilon-closure advance (the ALL next-configs, the no the single-config
    /// advance_eps). From the input config-set, follow the epsilon closure of each config +
    /// collect EVERY terminal-`a` move (the no the first). This is the sound step for a
    /// non-deterministic PDA (the one_or_more loop keeps the loop branch alive, the no the
    /// base-case-only advance that advance_eps gives). Returns the deduped next-config set.
    pub fn advance_eps_set(
        &self,
        configs: &[(u32, Vec<u32>)],
        a: u32,
    ) -> Vec<(u32, Vec<u32>)> {
        let use_csr = !self.ctrl_offsets.is_empty();
        let mut result: Vec<(u32, Vec<u32>)> = Vec::new();
        for (q, stk) in configs {
            // The epsilon closure of (q, stk). The dedup is on the FULL (state, stack) config (the no the
            // (state, top) dedup, which collapses distinct one_or_more loop iterations that share a
            // (state, top) but differ in stack depth, and thereby drops the loop-continue branch).
            // The full-stack key is lossless: two configs with the same state but different stacks
            // have different futures. The closure is finite for a bounded machine (the stack depth
            // is bounded by the machine's push structure), so the BFS terminates without a cap.
            let mut closed: Vec<(u32, Vec<u32>)> = vec![(*q, stk.clone())];
            let mut seen: HashSet<(u32, Vec<u32>)> = HashSet::new();
            seen.insert((*q, stk.clone()));
            // The stack-depth bound for the closure: the advance_eps_set is the NPDA frontier step (the the
            // loop preservation), so it needs the FULL depth (the num_states, the the pigeonhole) to
            // preserve the one_or_more re-entry (the the stack grows by one return-address per
            // iteration, the the no the tight max_stack_depth which truncates the loop). The
            // mask_at_cfg (the the mask, the the bounded) uses the tight closure_depth_bound (the
            // the VPA max_stack_depth), but the advance_eps_set (the the frontier, the the loop)
            // uses the num_states (the the full depth, the the loop preservation).
            let depth_bound = self.num_states as usize;
            let mut i = 0;
            while i < closed.len() {
                let (cq, cstk) = closed[i].clone();
                i += 1;
                if cstk.len() > depth_bound {
                    continue; // beyond the machine's reachable depth (the no the unbounded re-entry)
                }
                let ctop = cstk.last().copied().unwrap_or(self.start_stack);
                if use_csr {
                    let start = self.ctrl_offsets.get(cq as usize).copied().unwrap_or(self.transitions.len() as u32) as usize;
                    let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0) as usize;
                    for j in 0..count {
                        let t = &self.transitions[start + j];
                        if t.a != self.num_inputs || t.top != ctop {
                            continue;
                        }
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in t.push.iter().rev() {
                            ns.push(p);
                        }
                        if seen.insert((t.next_q, ns.clone())) {
                            closed.push((t.next_q, ns));
                        }
                    }
                } else {
                    for (q2, push) in self.transition(cq, None, ctop) {
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in push.iter().rev() {
                            ns.push(p);
                        }
                        if seen.insert((q2, ns.clone())) {
                            closed.push((q2, ns));
                        }
                    }
                }
            }
            // The terminal moves: collect ALL (the no the first) from every closed config.
            for (cq, cstk) in &closed {
                let top = cstk.last().copied().unwrap_or(self.start_stack);
                if use_csr {
                    let start = self.ctrl_offsets.get(*cq as usize).copied().unwrap_or(self.transitions.len() as u32) as usize;
                    let count = self.ctrl_counts.get(*cq as usize).copied().unwrap_or(0) as usize;
                    for j in 0..count {
                        let t = &self.transitions[start + j];
                        if t.a == a && t.top == top {
                            let mut s2 = cstk.clone();
                            s2.pop();
                            for &p in t.push.iter().rev() {
                                s2.push(p);
                            }
                            let new_cfg = (t.next_q, s2);
                            if !result.contains(&new_cfg) {
                                result.push(new_cfg);
                            }
                        }
                    }
                } else {
                    for t in self.lookup(*cq, Some(a), top) {
                        let mut s2 = cstk.clone();
                        s2.pop();
                        for &p in t.push.iter().rev() {
                            s2.push(p);
                        }
                        let new_cfg = (t.next_q, s2);
                        if !result.contains(&new_cfg) {
                            result.push(new_cfg);
                        }
                    }
                }
            }
        }
        result
    }

    /// The universal accepts: auto-selects the right simulation based on the
    /// machine's determinism. Callers do NOT need to know if the machine is a
    /// DPDA or an NPDA - this picks the correct one. It uses the deterministic
    /// path (the accepts_dpda, the single path, the fast) or the
    /// non-deterministic path (the accepts_npda, the BFS, the bounded).
    /// This is the "stupid-LLM-proof" entry point (the one method to call).
    pub fn accepts(&self, cfg: &Cfg, w: &[u32]) -> bool {
        if self.is_deterministic() {
            self.accepts_dpda(w)
        } else {
            self.accepts_npda(cfg, w)
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
    fn accepts_npda(&self, cfg: &Cfg, w: &[u32]) -> bool {
        // The NPDA acceptance is the CFG word problem (the L(M) == L(G), the RTN
        // compilation preserves the language). It is decided by the CYK algorithm
        // (the O(n^3) decidable word problem for context-free languages, the
        // Hopcroft-Ullman), NOT by the PDA frontier simulation (the exponential,
        // the unbounded stack). The `self` (the PdaMachine) is the compiled product;
        // the acceptance is a property of the CFG (the source).
        // The w is in the LOCAL PDA input IDs (the 0..num_inputs). The CYK needs
        // the GLOBAL CFG symbol IDs (the terminals are num_nonterminals..+num_terminals).
        // Convert local -> global (the + num_nonterminals).
        let w_global: Vec<u32> = w.iter().map(|&x| x + cfg.num_nonterminals).collect();
        crate::compile::cyk_accepts(cfg, &w_global)
    }
}

// TODO: remove_for_production - the debug trace that decomposes the closure
// states step-by-step (the frontier after each epsilon-closure + input step).
impl PdaMachine {
    pub fn trace_npda(&self, w: &[u32]) {
        use HashSet;
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
                        if seen.insert((q2, s2.clone())) {
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
                    if seen.insert((q2, s2.clone())) {
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

    fn step_batch(&self, batch: &[(Self::Config, u32)]) -> Vec<Vec<Self::Config>> {
        // The FULL-DOMAIN per-step advance (the the advance_eps_set, the config-set
        // epsilon-closure + the terminal move, the SET of next-configs per input).
        // This is exact for BOTH machines (the deterministic produces a 1-element set,
        // the non-deterministic produces the full frontier, the no the dropped loop
        // branch). The SIMD (_batch (the simd_pipeline.rs) calls the SAME
        // advance_eps_set, so the SIMD == the scalar (the same sets, the no the
        // interface-dependent behavior).
        batch
            .iter()
            .map(|((q, stk), a)| {
                // The advance_eps_set over the single input config (q, stk) with the
                // token a: the SET of next-configs (the full domain, the no the
                // single-config advance_eps). On divergence (the empty set, the no
                // next-config), HOLD the input config (q, stk) (the the reject path,
                // the consistent with the step_batch_simd's hold semantics).
                let set = self.advance_eps_set(&[(*q, stk.clone())], *a);
                if set.is_empty() {
                    vec![(*q, stk.clone())] // the hold (the no transition, the reject path)
                } else {
                    set
                }
            })
            .collect()
    }

    fn mask_batch(&self, configs: &[Self::Config]) -> Vec<Self::Mask> {
        // Delegate to the full-stack mask_at_cfg (the consistent epsilon closure, the no the
        // single-top approximation that desyncs on the empty-push epsilon moves).
        configs
            .iter()
            .map(|(q, stk)| self.mask_at_cfg(*q, stk))
            .collect()
    }

    fn project_batch(&self, configs: &[Self::Config], drafts: &[Vec<u32>]) -> Vec<Vec<Self::Mask>> {
        configs
            .iter()
            .zip(drafts.iter())
            .map(|((q0, stk0), draft)| {
                // The full-domain projection: the draft walk tracks the config-SET
                // (the frontier, the advance_eps_set, the no the single-config
                // advance_eps which drops the loop branch). The mask at each
                // position is the UNION over the frontier (the mask_at_cfg on each
                // frontier config, the deduped + the sorted).
                let mut frontier: Vec<(u32, Vec<u32>)> = vec![(*q0, stk0.clone())];
                let mut masks = vec![self.frontier_mask(&frontier)];
                for &a in draft {
                    frontier = self.advance_eps_set(&frontier, a);
                    if frontier.is_empty() {
                        break; // the draft diverged (the no next-config, the exclusive stop)
                    }
                    masks.push(self.frontier_mask(&frontier));
                }
                masks
            })
            .collect()
    }
}

impl PdaMachine {
    /// The mask over a config-SET (the frontier): the union of the mask_at_cfg over
    /// every config in the set (the deduped + the sorted, the no the order-
    /// dependent). This is the full-domain per-step mask (the the advance_eps_set
    /// frontier, the no the single-config).
    pub fn frontier_mask(&self, frontier: &[(u32, Vec<u32>)]) -> Vec<u32> {
        let mut allowed: HashSet<u32> = HashSet::new();
        for (q, stk) in frontier {
            for a in self.mask_at_cfg(*q, stk) {
                allowed.insert(a);
            }
        }
        let mut result: Vec<u32> = allowed.into_iter().collect();
        result.sort();
        result
    }

    /// The mask at a single config (the (state, stack)): the epsilon-closure union
    /// of the allowed inputs (the follow the epsilon moves to every reachable
    /// config, then collect the terminals with a defined move). This is the
    /// single-config entry for the PDA-as-FSM-mirror contract — it is exactly
    /// the set of inputs for which `advance_eps` succeeds (the proof
    /// `proof_mask_batch_consistent_with_advance_eps`). `mask_batch` is the
    /// batched form of this same computation.
pub fn mask_at_cfg(&self, q: u32, stack: &[u32]) -> Vec<u32> {
        // The precomputed epsilon-closure lookup (the the owl automaton_compute_epsilon_closure
        // pattern, the the O(1) replacement for the per-call BFS). The closure table is computed
        // once at construction (the compute_closure_table), and looked up here in O(1).
        //
        // The owl prior art: the owl interpreter precomputes the epsilon closure per state
        // (the epsilon_closure_for_state[i]), and the follow_transition step is O(1) (the the
        // no the per-step closure recomputation). My transcription: the closure is per (state, top)
        // pair (the the PDA's config space), and the mask_at_cfg looks up the closure in O(1).
        //
        // The soundness: the closure table is the least fixed point of the epsilon relation (the
        // the Bouajjani 1997 summarization), precomputed over the finite (state, top) domain
        // (the num_states x num_stack_syms). The GreatGramma Prop 3.5 (the stack invariance)
        // guarantees the (state, top) dedup is sound for the no-growth-cycle machines (the VPA /
        // the bounded-nesting). For the growth-cycle machines (the the star-loop), the (state, top)
        // dedup is NOT sound (the the cycle keeps the top constant while the stack grows, so the
        // terminal states reachable only at a deeper stack are missed). The growth-cycle machines
        // use the per-call full-stack BFS (the the no the precomputed table).
        if !self.has_epsilon_growth_cycle() && !self.closure_table.is_empty() {
            // The precomputed table (the the O(1) lookup, the the no-growth-cycle machines).
            // The table is empty for the direct-construction sites (the the tests, the the
            // examples) that don't call the compute_closure_table — fall back to the per-call BFS.
            let top = stack.last().copied().unwrap_or(self.start_stack);
            let closure = self.closure_at(q, top);
            // Collect the allowed inputs over the closure (the the O(closure_width) scan, the the
            // no the per-call BFS).
            let mut allowed: HashSet<u32> = HashSet::new();
            let use_csr = !self.ctrl_offsets.is_empty();
            for &(cq, ctop) in closure {
                if use_csr {
                    let start = self
                        .ctrl_offsets
                        .get(cq as usize)
                        .copied()
                        .unwrap_or(self.transitions.len() as u32) as usize;
                    let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0) as usize;
                    for j in 0..count {
                        let t = &self.transitions[start + j];
                        if t.a < self.num_inputs && t.top == ctop {
                            allowed.insert(t.a);
                        }
                    }
                } else {
                    for a in 0..self.num_inputs {
                        if !self.lookup(cq, Some(a), ctop).is_empty() {
                            allowed.insert(a);
                        }
                    }
                }
            }
            let mut result: Vec<u32> = allowed.into_iter().collect();
            result.sort();
            return result;
        }
        // The growth-cycle machines (the the star-loop): the per-call full-stack BFS (the the
        // no the precomputed table, the the (state, top) dedup is not sound for the growth cycle).
        let full_stack = true;
        let depth_bound = self.closure_depth_bound();
        let mut closed: Vec<(u32, Vec<u32>)> = vec![(q, stack.to_vec())];
        let use_csr = !self.ctrl_offsets.is_empty();
        if full_stack {
            // The full-stack dedup (the NPDA frontier, the lossless).
            let mut seen: HashSet<(u32, Vec<u32>)> = HashSet::new();
            seen.insert((q, stack.to_vec()));
            let mut i = 0;
            while i < closed.len() {
                let (cq, cstk) = closed[i].clone();
                i += 1;
                if cstk.len() > depth_bound {
                    continue; // beyond the machine's reachable depth (the pigeonhole bound)
                }
                let ctop = cstk.last().copied().unwrap_or(self.start_stack);
                if use_csr {
                    let start = self
                        .ctrl_offsets
                        .get(cq as usize)
                        .copied()
                        .unwrap_or(self.transitions.len() as u32) as usize;
                    let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0) as usize;
                    for j in 0..count {
                        let t = &self.transitions[start + j];
                        if t.a != self.num_inputs || t.top != ctop {
                            continue;
                        }
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in t.push.iter().rev() {
                            ns.push(p);
                        }
                        if seen.insert((t.next_q, ns.clone())) {
                            closed.push((t.next_q, ns));
                        }
                    }
                } else {
                    for (q2, push) in self.transition(cq, None, ctop) {
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in push.iter().rev() {
                            ns.push(p);
                        }
                        if seen.insert((q2, ns.clone())) {
                            closed.push((q2, ns));
                        }
                    }
                }
            }
        }
        // Collect the allowed inputs over the closure.
        let mut allowed: HashSet<u32> = HashSet::new();
        for (cq, cstk) in &closed {
            let ctop = cstk.last().copied().unwrap_or(self.start_stack);
            if use_csr {
                let start = self
                    .ctrl_offsets
                    .get(*cq as usize)
                    .copied()
                    .unwrap_or(self.transitions.len() as u32) as usize;
                let count = self.ctrl_counts.get(*cq as usize).copied().unwrap_or(0) as usize;
                for j in 0..count {
                    let t = &self.transitions[start + j];
                    if t.a < self.num_inputs && t.top == ctop {
                        allowed.insert(t.a);
                    }
                }
            } else {
                for a in 0..self.num_inputs {
                    if !self.lookup(*cq, Some(a), ctop).is_empty() {
                        allowed.insert(a);
                    }
                }
            }
        }
        let mut result: Vec<u32> = allowed.into_iter().collect();
        result.sort();
        result
    }

    /// The maximum epsilon-closure width (the distinct (state, top) pairs reachable) over ALL
    /// start configs (q, top). This sizes the GPU scan_mask / advance_pda BFS frontier (the
    /// MAXF): the frontier must hold at least this many pairs or the closure (and hence the
    /// mask) is truncated. O(num_states x num_stack_syms x closure_cost).
    pub fn max_closure_width(&self) -> usize {
        let mut max_width = 0usize;
        for q in 0..self.num_states {
            for top in 0..self.num_stack_syms {
                let stack = vec![top];
                // Reuse mask_at_cfg's closure by counting its distinct (state, top) visits via a
                // local BFS mirror (the mask_at_cfg returns the allowed inputs, not the width).
                let mut visited: HashSet<(u32, u32)> = HashSet::new();
                let mut frontier: Vec<(u32, u32)> = vec![(q, top)];
                let use_csr = !self.ctrl_offsets.is_empty();
                while let Some((cq, ctop)) = frontier.pop() {
                    if !visited.insert((cq, ctop)) {
                        continue;
                    }
                    if use_csr {
                        let start = self
                            .ctrl_offsets
                            .get(cq as usize)
                            .copied()
                            .unwrap_or(self.transitions.len() as u32) as usize;
                        let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0) as usize;
                        for t in &self.transitions[start..start + count] {
                            if t.top != ctop {
                                continue;
                            }
                            if t.a == self.num_inputs {
                                let new_top = if t.push.is_empty() {
                                    ctop
                                } else {
                                    *t.push.first().unwrap()
                                };
                                frontier.push((t.next_q, new_top));
                            }
                        }
                    } else {
                        for (q2, push) in self.transition(cq, None, ctop) {
                            let new_top = if push.is_empty() {
                                ctop
                            } else {
                                *push.first().unwrap()
                            };
                            frontier.push((q2, new_top));
                        }
                    }
                }
                let _ = &stack;
                max_width = max_width.max(visited.len());
            }
        }
        max_width
    }

    /// Whether the state `q` has an epsilon successor at stack-top `top` (the the choice/call/return/
    /// exit move, the the state-driven epsilon). For the VPA fast-path in mask_at_cfg: if the state
    /// has NO epsilon successor, it is at a settled config (the the O(1-3) mask_at_cfg_settled is
    /// the exact mask, the no the closure BFS). If it HAS an epsilon successor (the the choice state,
    /// the the call-dot), the closure is needed (the the fall through to the BFS).
    ///
    /// O(counts[q]) via the CSR (the the sorted-by-q array); the linear fallback for the hand-built
    /// machines (the no CSR).
    pub fn has_epsilon_successor(&self, q: u32, top: u32) -> bool {
        if self.ctrl_offsets.is_empty() {
            // The linear fallback (the the hand-built machine, the no CSR).
            return self
                .transitions
                .iter()
                .any(|t| t.q == q && t.a == self.num_inputs && t.top == top);
        }
        // The CSR fast path: scan only q's transitions (the O(counts[q]), the the sorted-by-q array).
        let start = self
            .ctrl_offsets
            .get(q as usize)
            .copied()
            .unwrap_or(self.transitions.len() as u32) as usize;
        let count = self.ctrl_counts.get(q as usize).copied().unwrap_or(0) as usize;
        for t in &self.transitions[start..start + count] {
            if t.a == self.num_inputs && t.top == top {
                return true;
            }
        }
        false
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
        if self.ctrl_offsets.is_empty() {
            // The linear fallback (the hand-built machine, the no CSR): scan all
            // inputs via the order-independent `lookup`.
            for a in 0..self.num_inputs {
                if !self.lookup(q, Some(a), top).is_empty() {
                    allowed.push(a);
                }
            }
        } else {
            // The CSR fast path: scan only q's transitions (the O(counts[q]),
            // the sorted-by-q array). Collect the inputs with a defined move at
            // (q, a, top). This is the O(1-3) replacement for the O(num_inputs
            // x total_transitions) linear scan.
            let start = self
                .ctrl_offsets
                .get(q as usize)
                .copied()
                .unwrap_or(self.transitions.len() as u32) as usize;
            let count = self.ctrl_counts.get(q as usize).copied().unwrap_or(0) as usize;
            for t in &self.transitions[start..start + count] {
                if t.top == top && t.a < self.num_inputs && !allowed.contains(&t.a) {
                    allowed.push(t.a);
                }
            }
        }
        allowed
    }

    /// Whether the config (q, stack) can reach an accepting state via epsilon
    /// moves only (the no input consumed). This is the PDA's final-state
    /// acceptance criterion at the config level: the PDA has "finished" the
    /// grammar iff the epsilon-closure of the current config contains an
    /// accepting state.
    ///
    /// This is the correct check for the EOS (the end-of-sequence) token: after
    /// consuming the last grammar token, the PDA may be at a dot state (the
    /// q_(p,m)), and the accepting state (the q_out(start)) is reached only via
    /// the exit epsilon move. Checking `accepting.contains(q)` (the is q itself
    /// accepting) would reject the EOS in that case. `accepts_via_eps` follows
    /// the epsilon closure and accepts correctly.
    ///
    /// The stack is tracked in full (the call pushes, the return pops), so the
    /// closure is over the (q, stack) configs (the bounded by the D, bound).
    /// O(epsilon_closure) via the CSR (the sorted-by-q array); the linear
    /// fallback for the hand-built machines (the no CSR).
    pub fn accepts_via_eps(&self, q: u32, stack: &[u32]) -> bool {
        // The full-stack (state, Vec) dedup for ALL machines (the the no the (state, top)
        // dedup, which truncates the closure for the growth-cycle machines). The num_states
        // depth bound (the the pigeonhole) ensures termination. This matches the original
        // c6646da behavior (the the full-stack dedup, the the unbounded exploration) with the
        // added depth bound (the the no the hang).
        let mut visited: HashSet<(u32, Vec<u32>)> = HashSet::new();
        let mut frontier: Vec<(u32, Vec<u32>)> = vec![(q, stack.to_vec())];
        let use_csr = !self.ctrl_offsets.is_empty();
        let depth_bound = self.num_states as usize;
        visited.insert((q, stack.to_vec()));
        while let Some((cq, cstk)) = frontier.pop() {
            if cstk.len() > depth_bound {
                continue;
            }
            if self.accepting.contains(&cq) {
                return true;
            }
            let ctop = cstk.last().copied().unwrap_or(self.start_stack);
            if use_csr {
                let start = self.ctrl_offsets.get(cq as usize).copied().unwrap_or(self.transitions.len() as u32) as usize;
                let count = self.ctrl_counts.get(cq as usize).copied().unwrap_or(0) as usize;
                for t in &self.transitions[start..start + count] {
                    if t.top != ctop || t.a < self.num_inputs {
                        continue;
                    }
                    let mut ns = cstk.clone();
                    ns.pop();
                    for &p in t.push.iter().rev() {
                        ns.push(p);
                    }
                    if visited.insert((t.next_q, ns.clone())) {
                        frontier.push((t.next_q, ns));
                    }
                }
            } else {
                for (q2, push) in self.transition(cq, None, ctop) {
                    let mut ns = cstk.clone();
                    ns.pop();
                    for &p in push.iter().rev() {
                        ns.push(p);
                    }
                    if visited.insert((q2, ns.clone())) {
                        frontier.push((q2, ns));
                    }
                }
            }
        }
        false
    }

    /// The pass-through predicate. True iff the input-consuming transition at config    /// config (q, top) for input `a` preserves the stack top (the push == [top]):
    /// a "shift" that does not modify the stack. In the RTN construction these are
    /// exactly the terminal moves (the dot advances within a production, the phase
    /// is invariant). This is the complement of the control edges (the call /
    /// return / choice / exit, which push, pop, or branch).
    ///
    /// O(1-3) via the CSR (the sorted-by-q array); the linear fallback for the
    /// hand-built machines (the no CSR).
    pub fn is_passthrough(&self, q: u32, top: u32, a: u32) -> bool {
        if a >= self.num_inputs {
            return false; // the epsilon is not a pass-through (the no input consumed)
        }
        if self.ctrl_offsets.is_empty() {
            // The linear fallback: scan all of q's transitions.
            return self
                .transitions
                .iter()
                .any(|t| t.q == q && t.a == a && t.top == top && t.push.len() == 1 && t.push[0] == top);
        }
        let start = self
            .ctrl_offsets
            .get(q as usize)
            .copied()
            .unwrap_or(self.transitions.len() as u32) as usize;
        let count = self.ctrl_counts.get(q as usize).copied().unwrap_or(0) as usize;
        for t in &self.transitions[start..start + count] {
            if t.a == a && t.top == top && t.push.len() == 1 && t.push[0] == top {
                return true;
            }
        }
        false
    }

    /// The next control state via the input-consuming pass-through shift from `q`
    /// (the stack-preserving terminal move), or None when there is no such move
    /// (a control edge, or an epsilon-only state). O(counts[q]) via the CSR.
    fn passthrough_next(&self, q: u32) -> Option<u32> {
        if self.ctrl_offsets.is_empty() {
            for t in self.transitions.iter().filter(|t| t.q == q) {
                if t.a < self.num_inputs && t.push.len() == 1 && t.push[0] == t.top {
                    return Some(t.next_q);
                }
            }
            return None;
        }
        let start = self
            .ctrl_offsets
            .get(q as usize)
            .copied()
            .unwrap_or(self.transitions.len() as u32) as usize;
        let count = self.ctrl_counts.get(q as usize).copied().unwrap_or(0) as usize;
        for t in &self.transitions[start..start + count] {
            if t.a < self.num_inputs && t.push.len() == 1 && t.push[0] == t.top {
                return Some(t.next_q);
            }
        }
        None
    }

    /// The linear-run length from control state `q`: the number of consecutive
    /// pass-through (the stack-preserving) shifts before the next control edge
    /// (the call / return / choice / exit). In the RTN construction this is the
    /// run of consecutive terminals in the production's rhs starting at the dot
    /// (the top-independent: the terminal shifts preserve the stack top).
    ///
    /// The run is bounded by the production length (the six-property "bounded
    /// control"), never by the input length. O(run x counts[q]) via the CSR.
    pub fn passthrough_run(&self, q: u32) -> u32 {
        let cap = self.num_states; // the safety cap (the no infinite loop)
        let mut cur = q;
        let mut depth = 0u32;
        while depth < cap {
            match self.passthrough_next(cur) {
                Some(nq) => {
                    cur = nq;
                    depth += 1;
                }
                None => break,
            }
        }
        depth
    }

    /// The displacement of a terminal sequence `t` (the CFGzip Theorem 2 primitive):
    /// the set of (in_config, out_config) pairs such that out_config is reachable
    /// from in_config by consuming `t` (via the PDA's transition function). This is
    /// the pure, context-free stack-transformation function that defines the token
    /// equivalence classes (the displacement partition): two tokens are
    /// interchangeable iff they have the same displacement.
    ///
    /// The config is the (control_state, stack) pair. The displacement is computed
    /// by simulating the PDA over `t` from every reachable in_config (the bounded
    /// stack, the six-property "bounded pushdown"). This is a pure function (the no
    /// temporary state approximating the math): the input is the terminal sequence,
    /// the output is the set of (in_config, out_config) pairs.
    ///
    /// O(reachable_configs x |t| x transitions[q]) via the CSR (the sorted-by-q array).
    pub fn displacement(&self, t: &[u32]) -> Vec<(u32, Vec<u32>, u32, Vec<u32>)> {
        let in_configs = self.reachable_in_configs();
        self.displacement_from(&in_configs, t)
    }

    /// The reachable in_configs (the BFS over the PDA's input closure from the start
    /// config). This is the expensive, machine-invariant part of the displacement: it
    /// depends only on the machine, not on the input sequence, so it is computed ONCE
    /// and reused across many displacement queries (the displacement_many).
    ///
    /// PRECONDITION (the six-property #5, the bounded pushdown): the machine's stack is
    /// bounded by D = max_push + 1. The BFS is finite (the num_states * the
    /// num_stack_syms^D configs), so it terminates. The guard: if the BFS exceeds the
    /// finite domain size, the machine is NOT six-property #5 bounded (the unbounded
    /// stack), and the displacement is undefined (the explicit panic, the no the silent
    /// non-termination).
    pub fn reachable_in_configs(&self) -> Vec<(u32, Vec<u32>)> {
        // The reachable (state, full-stack) in_configs, by a SINGLE worklist BFS over the
        // machine's moves. A config is reached by consuming input; the epsilon moves are the
        // a == num_inputs moves, so one uniform successor walk over all (num_inputs + 1) input
        // values per config is exactly the reference reachability (the epsilon steps + the
        // epsilon prefixes folded in) in ONE pass (the no the per-(config x input) re-BFS that
        // was O(reachable x inputs x closure) and locked up every CPU on the Qwen machine).
        //
        // The dedup is on the FULL (state, stack) config (the no the (state, top) dedup, which
        // collapses distinct stacks that share a top and loses the displacement's in_config
        // distinction). The domain is finite for a bounded-pushdown machine (the six-property
        // #5), so the BFS terminates; the guard panics (the no the silent non-termination) if it
        // does not.
        let use_csr = !self.ctrl_offsets.is_empty();
        // The depth cap for termination: a reachable stack deeper than num_states must repeat a
        // (state, top) pair (the pigeonhole principle over the num_states control states), hence
        // is a cycle and can be pruned without losing any reachable in_config. This is what makes
        // the full-stack BFS terminate on the one_or_more re-entry (which otherwise grows the
        // stack without bound). Derived from the machine's own num_states (the no a synthetic
        // constant).
        let depth_cap = self.num_states as usize;
        let mut in_configs: Vec<(u32, Vec<u32>)> = vec![(self.start_state, vec![self.start_stack])];
        // The outer worklist dedups on the FULL (state, stack) config (the no the (state, top)
        // dedup, which collapses distinct loop iterations that share a (state, top) but differ in
        // stack depth, and thereby drops the one_or_more loop-continue branch the displacement
        // needs). Termination comes from the depth_cap above.
        let mut seen: std::collections::HashSet<(u32, Vec<u32>)> =
            std::collections::HashSet::from([(self.start_state, vec![self.start_stack])]);
        let mut i = 0;
        while i < in_configs.len() {
            let (cq, cstk) = in_configs[i].clone();
            i += 1;
            // The epsilon-closure of (cq, cstk): follow ONLY the epsilon moves (the a ==
            // num_inputs), deduping on the FULL (state, stack) config (the no the (state, top)
            // dedup, which collapses distinct loop iterations that share a (state, top) but
            // differ in stack depth, and thereby drops the one_or_more loop-continue branch).
            // Bounded by depth_cap so the one_or_more re-entry (which grows the stack) terminates.
            let mut eps: Vec<(u32, Vec<u32>)> = vec![(cq, cstk.clone())];
            let mut eps_seen: std::collections::HashSet<(u32, Vec<u32>)> =
                std::collections::HashSet::new();
            eps_seen.insert((cq, cstk.clone()));
            let mut ei = 0;
            while ei < eps.len() {
                let (eq, estack) = eps[ei].clone();
                ei += 1;
                let etop = estack.last().copied().unwrap_or(self.start_stack);
                let eps_succs: Vec<(u32, Vec<u32>)> = if use_csr {
                    let start = self
                        .ctrl_offsets
                        .get(eq as usize)
                        .copied()
                        .unwrap_or(self.transitions.len() as u32) as usize;
                    let count = self.ctrl_counts.get(eq as usize).copied().unwrap_or(0) as usize;
                    self.transitions[start..start + count]
                        .iter()
                        .filter(|t| t.a == self.num_inputs && t.top == etop)
                        .map(|t| {
                            let mut ns = estack.clone();
                            ns.pop();
                            for &p in t.push.iter().rev() {
                                ns.push(p);
                            }
                            (t.next_q, ns)
                        })
                        .collect()
                } else {
                    self.transition(eq, None, etop)
                        .into_iter()
                        .map(|(q2, push)| {
                            let mut ns = estack.clone();
                            ns.pop();
                            for &p in push.iter().rev() {
                                ns.push(p);
                            }
                            (q2, ns)
                        })
                        .collect()
                };
                for (nq, ns) in eps_succs {
                    // The depth cap: a stack deeper than num_states is a cycle (the pigeonhole
                    // principle), safe to prune without losing any reachable in_config.
                    if ns.len() > depth_cap {
                        continue;
                    }
                    if eps_seen.insert((nq, ns.clone())) {
                        eps.push((nq, ns));
                    }
                }
            }
            // The input-successors: from every epsilon-closed config, take every INPUT move (the
            // a in 0..num_inputs). These enqueue new in_configs (the epsilon moves are traversed
            // only as the prefix of an input step).
            for (eq, estack) in &eps {
                let etop = estack.last().copied().unwrap_or(self.start_stack);
                let in_succs: Vec<(u32, Vec<u32>)> = if use_csr {
                    let start = self
                        .ctrl_offsets
                        .get(*eq as usize)
                        .copied()
                        .unwrap_or(self.transitions.len() as u32) as usize;
                    let count = self.ctrl_counts.get(*eq as usize).copied().unwrap_or(0) as usize;
                    self.transitions[start..start + count]
                        .iter()
                        .filter(|t| t.a < self.num_inputs && t.top == etop)
                        .map(|t| {
                            let mut ns = estack.clone();
                            ns.pop();
                            for &p in t.push.iter().rev() {
                                ns.push(p);
                            }
                            (t.next_q, ns)
                        })
                        .collect()
                } else {
                    (0..self.num_inputs)
                        .flat_map(|a| self.lookup(*eq, Some(a), etop))
                        .map(|t| {
                            let mut ns = estack.clone();
                            ns.pop();
                            for &p in t.push.iter().rev() {
                                ns.push(p);
                            }
                            (t.next_q, ns)
                        })
                        .collect()
                };
                for (nq, ns) in in_succs {
                    // The depth cap: a stack deeper than num_states is a cycle (the pigeonhole
                    // principle), safe to prune without losing any reachable in_config.
                    if ns.len() > depth_cap {
                        continue;
                    }
                    if seen.insert((nq, ns.clone())) {
                        in_configs.push((nq, ns));
                    }
                }
            }
        }
        in_configs
    }

    /// The displacement of a terminal sequence `t` from a FIXED set of in_configs (the
    /// the no set, the no the per-call BFS). This is the cheap per-sequence part:
    /// simulate the PDA over `t` from every in_config and collect the (in, out) pairs.
    fn displacement_from(
        &self,
        in_configs: &[(u32, Vec<u32>)],
        t: &[u32],
    ) -> Vec<(u32, Vec<u32>, u32, Vec<u32>)> {
        let mut result: Vec<(u32, Vec<u32>, u32, Vec<u32>)> = Vec::new();
        for (in_q, in_stack) in in_configs {
            // Simulate the PDA over t from this in_config, keepingING the full config-set (the
            // NPDA frontier) so the one_or_more loop branches are all preserved. Each step: the
            // epsilon-closure (full-stack, the no the (state, top) collapse that drops the
            // iterations) then ALL input-`a` moves.
            let mut configs: Vec<(u32, Vec<u32>)> = vec![(*in_q, in_stack.clone())];
            let mut diverged = false;
            for &a in t {
                // The epsilon-closure of the current frontier (the full-stack, the exact).
                let mut closed: Vec<(u32, Vec<u32>)> = configs.clone();
                let mut seen: std::collections::HashSet<(u32, Vec<u32>)> =
                    closed.iter().cloned().collect();
                let mut ci = 0;
                while ci < closed.len() {
                    let (cq, cstk) = closed[ci].clone();
                    ci += 1;
                    let ctop = cstk.last().copied().unwrap_or(self.start_stack);
                    for (q2, push) in self.transition(cq, None, ctop) {
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in push.iter().rev() {
                            ns.push(p);
                        }
                        if seen.insert((q2, ns.clone())) {
                            closed.push((q2, ns));
                        }
                    }
                }
                // The input-`a` moves from every closed config.
                let mut next: Vec<(u32, Vec<u32>)> = Vec::new();
                let mut next_seen: std::collections::HashSet<(u32, Vec<u32>)> = std::collections::HashSet::new();
                for (cq, cstk) in &closed {
                    let ctop = cstk.last().copied().unwrap_or(self.start_stack);
                    for (q2, push) in self.transition(*cq, Some(a), ctop) {
                        let mut ns = cstk.clone();
                        ns.pop();
                        for &p in push.iter().rev() {
                            ns.push(p);
                        }
                        if next_seen.insert((q2, ns.clone())) {
                            next.push((q2, ns));
                        }
                    }
                }
                if next.is_empty() {
                    diverged = true;
                    break; // the sequence diverged (the no out_config)
                }
                configs = next;
            }
            if !diverged {
                for (out_q, out_stack) in &configs {
                    result.push((*in_q, in_stack.clone(), *out_q, out_stack.clone()));
                }
            }
        }
        result
    }

    /// The displacement of MANY terminal sequences against a SINGLE shared reachable
    /// in_config set (the the form, the no the per-sequence BFS recomputation). This is
    /// the production primitive for the bridge (the vocab_size sequences, the one BFS).
    pub fn displacement_many(&self, sequences: &[Vec<u32>]) -> Vec<Vec<(u32, Vec<u32>, u32, Vec<u32>)>> {
        let in_configs = self.reachable_in_configs();
        // The per-sequence displacement is independent (a pure function of the sequence + the
        // fixed in_config set), so parallelize across sequences (the rayon, the no shared state).
        // This makes the one-time bridge build tractable on the 248K Qwen vocab.
        use rayon::prelude::*;
        sequences
            .par_iter()
            .map(|s| self.displacement_from(&in_configs, s))
            .collect()
    }

    /// The displacement partition (the CFGzip Theorem 2): group a set of terminal
    /// sequences by their displacement (the set of (in_config, out_config) pairs).
    /// Two sequences are in the same group iff they have the same displacement (the
    /// interchangeable tokens). This is the bridge (the terminal -> token map)
    /// computed via the displacement equivalence (the no DFA).
    pub fn displacement_partition(&self, sequences: &[Vec<u32>]) -> Vec<Vec<usize>> {
        // The displacement signature (the sorted set of (in_config, out_config) pairs)
        // for each sequence.
        let signatures: Vec<Vec<(u32, Vec<u32>, u32, Vec<u32>)>> =
            sequences.iter().map(|s| self.displacement(s)).collect();
        // Group the sequences by their signature (the displacement equivalence).
        let mut groups: Vec<(Vec<(u32, Vec<u32>, u32, Vec<u32>)>, Vec<usize>)> = Vec::new();
        for (idx, sig) in signatures.iter().enumerate() {
            match groups.iter_mut().find(|(g, _)| g == sig) {
                Some((_, ids)) => ids.push(idx),
                None => groups.push((sig.clone(), vec![idx])),
            }
        }
        groups.into_iter().map(|(_, ids)| ids).collect()
    }

    /// The displacement composition (the functional property): the displacement of
    /// the concatenation t1 ++ t2 is the composition of the displacements (the
    /// D(t1 ++ t2) = D(t2) D(t1), the t1 is consumed first, then the t2). This is
    /// the relation composition (the set of (in_config, out_config) pairs such that
    /// there exists an intermediate config).
    ///
    /// The `d_first` is the displacement of t1 (the consumed first), and the
    /// `d_second` is the displacement of t2 (the consumed second). The composition
    /// is the d_second o the d_first (the relation composition).
    pub fn displacement_compose(
        d_first: &[(u32, Vec<u32>, u32, Vec<u32>)],
        d_second: &[(u32, Vec<u32>, u32, Vec<u32>)],
    ) -> Vec<(u32, Vec<u32>, u32, Vec<u32>)> {
        // The relation composition: the (a_in_ctrl, a_in_stack, c_out_ctrl,
        // c_out_stack) quadruples such that there exists an intermediate
        // (b_ctrl, b_stack) with (a_in_ctrl, a_in_stack, b_ctrl, b_stack) in
        // d_first AND (b_ctrl, b_stack, c_out_ctrl, c_out_stack) in d_second.
        let mut result: Vec<(u32, Vec<u32>, u32, Vec<u32>)> = Vec::new();
        for &(a_in_ctrl, ref a_in_stack, b_ctrl, ref b_stack) in d_first {
            for &(c_in_ctrl, ref c_in_stack, c_out_ctrl, ref c_out_stack) in d_second {
                if b_ctrl == c_in_ctrl && b_stack == c_in_stack {
                    result.push((a_in_ctrl, a_in_stack.clone(), c_out_ctrl, c_out_stack.clone()));
                }
            }
        }
        result
    }
}
impl EpsilonPda for PdaMachine {}
impl FinalStatePda for PdaMachine {}
impl EmptyStackPda for PdaMachine {}
impl DisplacementPda for PdaMachine {
    fn is_passthrough(&self, q: u32, top: u32, a: u32) -> bool {
        self.is_passthrough(q, top, a)
    }
    fn passthrough_run(&self, q: u32) -> u32 {
        self.passthrough_run(q)
    }
    fn displacement(&self, t: &[u32]) -> Vec<(u32, Vec<u32>, u32, Vec<u32>)> {
        self.displacement(t)
    }
    fn displacement_partition(&self, sequences: &[Vec<u32>]) -> Vec<Vec<usize>> {
        self.displacement_partition(sequences)
    }
    fn accepts_via_eps(&self, q: u32, stack: &[u32]) -> bool {
        self.accepts_via_eps(q, stack)
    }
}

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
