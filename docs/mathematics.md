# The mathematics

The formal definitions + the proofs for the pushdown-rs core.

## The 7-tuple

A pushdown automaton is M = (Q, Sigma, Gamma, delta, q0, Z0, F):
- Q: the finite control states.
- Sigma: the input alphabet (the terminals).
- Gamma: the stack alphabet.
- delta: Q x (Sigma U {epsilon}) x Gamma -> P(Q x Gamma*), the transition
  relation. For a DPDA, |delta(q, a, top)| <= 1.
- q0: the start state.
- Z0: the start stack symbol.
- F: the accepting states.

A configuration is (q, w, gamma) where w is the unread input, gamma is the
stack (Z0 at the bottom). A step is (q, aw, X gamma) -> (q', w, gamma') when
(q', gamma') in delta(q, a, X).

## The RTN compilation (the CFG -> the PDA)

Given a CFG G = (N, Sigma, P, S), the recursive-transition-network
compilation yields a PDA M_G:

- The control states: Q = {q_start} U {q_A^in, q_A^out : A in N} U
  {q_(p,i) : p in P, i in 0..=|rhs(p)|}.
- The stack alphabet: Gamma = {bot} U {q_(p,i) : p in P, i} (the return
  addresses).
- The transitions:
  - the start: delta(q_start, eps, bot) = (q_S^in, [bot]).
  - the choice: delta(q_A^in, eps, gamma) = (q_(p,0), gamma) for each
    production p of A.
  - the terminal: if X_i in Sigma, delta(q_(p,i-1), X_i, gamma) = (q_(p,i),
    gamma).
  - the call: if X_i = B in N, delta(q_(p,i-1), eps, gamma) = (q_B^in,
    [q_(p,i), gamma]).
  - the return: delta(q_B^out, eps, r) = (r, []) for each return address r.
  - the exit: delta(q_(p,m), eps, gamma) = (q_A^out, gamma).
- The accepting: F = {q_S^out}.

The exact control-state count is kappa(G) = 1 + 2|N| + sum_p(|rhs(p)| + 1).

Theorem (correctness): L(M_G) = L(G). The RTN compilation preserves the
language: a string is accepted by the PDA iff it is derived by the CFG.

## The determinism

The PDA is deterministic (a DPDA) iff for every (q, a, top), |delta(q, a, top)|
<= 1, AND if delta(q, eps, top) is defined then delta(q, a, top) is undefined
for all a in Sigma. A CFG compiles to a DPDA iff it is an LR(1) grammar (the
DCFL).

The is_deterministic() check verifies the at-most-one condition. The
accepts_dpda runs the unique computation path (the deterministic PDA has a
single path, the Hopcroft-Ullman); the DCFL = the DPDA-recognizable result);
the accepts_npda runs the CYK (the O(n^3) CFG word problem, the no the
exponential frontier simulation). The proof_dpda_single_path_agrees_with_cyk
test verifies the deterministic case (the accepts_dpda == the CYK) + the
non-deterministic case (the accepts_dpda == false, the CYK == the cfg_accepts
oracle).

## The bounded stack

For a grammar with max production length L, the stack depth is bounded by D = L
+ 1 (the pending nesting). The DPDA is GPU-resident because D is a small
constant for the target grammars (the JSON, the tool envelopes, the network
headers). The FFI D-bound (the cuda.rs PdaSeqState stack[8], the pda_example.cu
D_MAX 8) is this constant = max_rhs+1 for the target grammars (the max_rhs <= 7,
the D = 8), NOT a magic number.

## The VPA fast-path (the the O(1-3) settled mask, the no the closure BFS)

The JSON / the well-nested grammars are VisiblyPushdown (the the stack-op is a
function of the input symbol, the the Madhusudan VPA property). For these, the
mask at a SETTLED config (the no pending epsilon moves) is EXACTLY the settled
gate (the mask_at_cfg_settled, the O(1-3) CSR lookup), the no the closure BFS.
The iterative closure BFS thrashes the CPU on every token for the VPA case (the
the 115µs/call vs the 57µs legacy, the the PDA is SLOWER than the fallback).

The `mask_at_cfg` dispatchpa fast-path: when the machine is classified as
`VisiblyPushdown` AND the current (state, top) has NO epsilon successor (the
`has_epsilon_successor` is false, the the settled config), the mask is the
O(1-3) `mask_at_cfg_settled` (the no the BFS). When the state HAS an epsilon
successor (the the choice state, the the call-dot), the closure is needed (the
the fall through to the (state, top) BFS, the the GreatGramma Prop 3.5 sound).

The `advance_eps_set` (the NPDA frontier step, the the loop preservation) uses
the FULL depth (the num_states, the the pigeonhole), the no the tight
`closure_depth_bound` (the the VPA max_stack_depth truncates the one_or_more
re-entry, the the test_qwen_plus_loop_back_edge regression). The mask_at_cfg
(the the mask, the the bounded) uses the tight `closure_depth_bound`; the
advance_eps_set (the the frontier, the the loop) uses the full_states.

CITATIONS:
- Madhusudan et al. (the Visibly Pushdown Automata, the the well-nested
  languages, the the O(1) mask at a settled config).
- GreatGramma (Park, Zhou, D'Antoni, arXiv:2502.05111) Prop 3.5 (the stack
  invariance, the the (state, top) dedup soundness for the VPA / the bounded).
- The six-property #5 (the bounded pushdown, the the max_stack_depth, the the
  tight_stateless_grammar.md).
- The pigeonhole principle (the the num_states depth bound for the growth cycle).

## The PdaKind dispatch (the programmatic type determination, the tight bound per kind)

The epsilon-closure BFS (the mask_at_cfg, the advance_eps) must terminate with a
synthetic cap, AND use the tightest dedup that is sound for the machine. The soundness
of the (state, top) dedup (the GreatGramma Prop 3.5 invariance: the mask depends only on
(state, top), not the deeper stack) holds ONLY when the machine has no epsilon growth
cycle. A star-loop (the one_or_more re-entry) has a growth cycle: the cycle keeps the
top constant while the stack grows, so the (state, top) dedup prematurely terminates the
BFS before reaching the terminal states that are only reachable at a deeper stack (the
mask is empty, the under-approximation, the draft acceptance breaks).

The dispatch is programmatic (the no the caller guessing the kind):

- `classify() -> PdaKind`: the most-specific-first determination. `is_visibly_pushdown`
  (the stack-op class is a function of the input symbol alone, the VPA property: for each
  input a, all transitions consuming a have the same push-length class — the call
  (push.len() >= 2), the return (push.len() == 0), the internal (push.len() == 1)) -> the
  VisiblyPushdown. Else `is_deterministic` (the no duplicate (q, a, top)) -> the
  Deterministic. Else the NonDeterministic.
- `has_epsilon_growth_cycle()`: the Tarjan SCC over the epsilon graph (the states = nodes,
  the epsilon moves = edges). A growth cycle exists iff an SCC of size > 1 contains a
  growth edge (the push.len() >= 2, the call that re-enters the SCC). This is the
  distinguishing property: the star-loop (the growth cycle) needs the full-stack dedup; the
  VPA / the bounded-nesting (the no growth cycle) the (state, top) dedup is sound.
- `closure_depth_bound()`: the tight bound per kind. VisiblyPushdown -> the
  max_stack_depth (the nesting depth, the bounded). Deterministic -> the max_stack_depth
  when no growth cycle, else num_states (the pigeonhole). NonDeterministic -> the num_states.

The mask_at_cfg / the advance_eps dispatch on `has_epsilon_growth_cycle` (the the-stack
(state, Vec) dedup when there is a growth cycle, the (state, top) dedup when there is not),
with the `closure_depth_bound` termination. The linear reference (the
mask_at_cfg_linear_reference) matches the dispatch, so the proof_mask_at_cfg_csr_equals_
linear (the CSR == the linear, the three-way identity) holds.

CITATIONS:
- GreatGramma (Park, Zhou, D'Antoni, arXiv:2502.05540) Prop 3.5 (the stack invariance,
  the (state, top) dedup soundness condition).
- The six-property #5 (the bounded pushdown, the finite stackless_grammar.md).
- The pigeonhole principle (the num_states depth bound for the growth cycle).
- Tarjan's SCC (the growth-cycle detection).

## The unbounded weighted-PDS distance (the S, the no the S_H)

The unbounded distance d(c) is the least fixed point of the distance equations
over the (min,+) dioid (the Reps 2003, the "min-plus semiring finds the
shortest trace"): d(c) = 0 if c is accepting, else d(c) = 1 + min over c -> c'
of d(c'). This is the shortest-path-to-acceptance over the config graph,
computed by the reverse BFS (the pre* saturation, the Bouajjani 1997).

The PDA transition semantics (the ODU CS390, the delta(q, a, top) = {(q',
push)}): the successor stack is cstk = push.reversed() ++ sigma' where sigma'
= sigma without its top (the popped symbol). So the reverse step recovers the
predecessor sigma = sigma' ++ [top] where sigma' = cstk[|push|..] and top is a
free variable in Gamma (the popped symbol, the no in cstk, the transition only
sees the top). The rule (t.q, a, top) -> (t.next_q, t.push) must match (the
t.top == top).

The stack depth is bounded by D = max_push + 1 (the six-property #5, the
bounded pushdown: the max production length + 1, the pending nesting). The
max_push is derived from the machine (the maximum length of any transition's
push string, the no a static value). The reverse BFS is restricted to stacks of
depth <= D (the no the unbounded exploration, which does not terminate for
machines with a bounded forward stack). The BoundedSummary::compute_unbounded
(the no the compute(machine, h)) uses this intrinsic D bound (the h = D, the no
usize::MAX).

The six-property #5 precondition guard: if the machine's forward Reach BFS
truncates any config at depth > D (the the stack is NOT bounded by D, the
unbounded-stack machine like the a^n b^n), the compute_unbounded PANICS
explicitly (the no the silent omission). The caller must use the
compute(machine, h) with a caller-supplied h for unbounded-stack machines.

CITATIONS:
- Reps, Schwoon, Jha, Melski, "Weighted Pushdown Systems and Their Application
  to Interprocedural Dataflow Analysis," SAS 2003 (the (min,+) dioid, the
  "min-plus semiring finds the shortest trace").
- Bouajjani et al. 1997 (the pushdown-system reachability, the pre* / the
  post* saturation, the least fixed point).
- Schlichtkrull et al. (the AFP Pushdown_Systems, the pre_star_exec, the
  formally verified PDS reachability).
- ODU CS390 (the PDA transition semantics, the delta(q, a, top) = {(q', push)},
  the successor stack equation).
- The six-property #5 (the bounded pushdown, the D = max_rhs + 1, the
  finite_stateless_grammar.md). The FFI D-bound (the cuda.rs PdaSeqState stack[8], the pda_example.cu
D_MAX 8) is this constant = max_rhs+1 for the target grammars (the max_rhs <= 7, the
D = 8), NOT a magic number.

## The epsilon-closure advance

The PDA's advance by a terminal must resolve the epsilon moves first (the RTN
choice/call/exit/return are epsilon). The advance_eps(q, stk, a) follows the
epsilon closure from (q, stk) (the BFS over the epsilon moves, the terminated
by the dedup on the (state, top) pairs, the no the MAX_EPS_CLOSURE cap) to the
configs where the terminal move a is available, then does the terminal move.
This is consistent with the mask (the mask_at_cfg is the epsilon-closure of the
allowed inputs): the advance reaches exactly the
states whose mask allows a. Without it, the PDA would get stuck at the call
dots (the no direct terminal move), inconsistent with the mask.

The step_batch and the project_batch use advance_eps (the no stuck call dots).

The CSR index (ctrl_offsets + ctrl_counts, computed at construction) makes
the advance_eps transition lookup O(1-3) per state (instead of O(total)
linear scan). The CSR groups transitions by their control state q, so the
epsilon-closure BFS and the terminal move only scan the transitions for the
current state (typically 1-3 for RTN-compiled machines). The GPU kernel
(the attention-rs PdaPushdownTable) uses the same CSR format (the
ctrl_offsets + ctrl_counts uploaded to the GPU), so the CPU and GPU
paths are consistent (the three-way identity, the P8 property).

## The state provenance

The RTN state -> nonterminal projection (the state_provenance, the Definition 5):
for each control state, the nonterminal index it belongs to (the q_start -> S,
the q_A^in/q_A^out -> A, the dot(p,i) -> lhs(p)). It is a total function of -> N
of length kappa(G). The phase-homology invariant: the label is invariant along
a terminal run inside one production, and changes exactly at the call/return
boundaries (the callee/caller non The consumer maps the nonterminal index to a
semantic region via the grammar's named nonterminals.

## The pass-through (the "no control edge")

A committed input a at config (q, top) is a PASS-THROUGH (the no control edge)
iff the transition (q, a, top) -> (q', push) is an identity-stack terminal
shift: a is a real input (a < num_inputs, the no epsilon) AND push == [top]
(the stack is unchanged). In the RTN construction this is exactly the terminal
move delta(q_(p,i), X_i, top) = (q_(p,i+1), [top]): the dot advances within a
production, the phase (the state_provenance) is invariant, and the stack top is
preserved. The control edges (the call / return / choice / exit) are the
complement: they are epsilon moves that push a return address, pop, or branch.

The LINEAR RUN from a control state q is the number of consecutive pass-through
shifts before the next control edge. In the RTN construction this is the run of
consecutive terminals in the production's rhs starting at the dot (the top-
independent: the terminal shifts preserve the stack top). The run is bounded by
the production length (the six-property "bounded control"), NEVER by the input
length. This is the generic automata property a consumer exploits: within a
linear run the machine's control is locally deterministic (the single shift
path), so a consumer can treat the run as a stable region (the mask is phase-
invariant) and defer any heavier per-step work until the next control edge.

Theorem (pass-through soundness): is_passthrough(q, top, a) holds iff an
identity-stack input-consuming transition (q, a, top) -> (q', [top]) exists.
Proof: the predicate scans q's transitions (the CSR range) for exactly that
record (the a < num_inputs, the push == [top]); The inclusive direction: such a
record is a pass-through. The exclusive direction: any non-pass-through (the
epsilon, the push != [top], the no record) fails the scan. QED. The oracle is
the independent linear scan (the proof_is_passthrough_sound).

Theorem (run exactness): passthrough_run(q) equals the number of consecutive
pass-through shifts from q. Proof: the chain follows the unique input-consuming
identity-stack edge per state (the deterministic RTN), counting until a control
edge (the no such edge). The chain is acyclic (the dot positions only advance
within a production), so the count terminates at the production end. QED. The
oracle is the independent linear chain-follow (the proof_passthrough_run_exact).

## The displacement (the CFGzip Theorem 2, the pure functional atom)

The displacement of an input sequence t is the set of (in_config, out_config)
pairs such that out_config is reachable from in_config by consuming t (via the
PDA's transition function). This is the pure, context-free stack-transformation
relation (the no temporary state approximating the math). Two sequences
sequences are interchangeable iff they have the same displacement (the
displacement equivalence refines the syntactic congruence, the CFGzip Theorem
2, the lossless compression).

Theorem (displacement composition): the displacement of the concatenation t1 ++
t2 is the composition of the displacements (the D(t1 ++ t2) = D(t2) o D(t1),
the t1 is consumed first, then the t2). Proof: the composition of the relations
(the set of (in, out) pairs such that there exists an intermediate config) is
associative. The displacement is the RELATION over ALL reachable configs (the
advance_eps_set, the config-set advance, the no the single deterministic path).
QED. The oracle is the independent direct computation (the
proof_displacement_composition).

The displacement partition (the bridge): group a set of input sequences by
their displacement (the set of (in_config, out_config) pairs). Two sequences
are in the same group iff they have the same displacement (the interchangeable
inputs). This is the bridge (the terminal -> the token map) computed via the
displacement equivalence (the no DFA).

## The displacement (the CFGzip Theorem 2, the pure functional atom)

The displacement of an input sequence t is the set of (in_config, out_config)
pairs such that out_config is reachable from in_config by consuming t (via the
PDA's transition function). This is the pure, context-free stack-transformation
relation (the no temporary state approximating the math). Two sequences are
interchangeable iff they have the same displacement (the displacement
equivalence refines the syntactic congruence, the CFGzip Theorem 2, the
lossless compression).

The displacement forms a MONOID under composition (the D(t1 ++ t2) = D(t2) o
D(t1)). The identity is the empty sequence (the D(()) = the identity relation,
the (c, c) pairs for all reachable configs c). The associativity is the
D(t1 ++ t2 ++ t3) = (D(t3) o D(t2)) o D(t1) = D(t3 ++ t2) o D(t1). This is the
key algebraic property that makes the displacement a functional atom (the no
temporary state, the pure function). The proofs are the
proof_displacement_monoid_identity + the proof_displacement_monoid_
associativity.

The displacement partition (the bridge): group a set of input sequences by
their displacement (the set of (in_config, out_config) pairs). Two sequences
are in the same group iff they have the same displacement (the interchangeable
inputs). This is the bridge (the terminal -> the token map) computed via the
displacement equivalence (the no DFA).

## The CSR sort invariant

The CSR index (the ctrl_offsets + the ctrl_counts, the first-occurrence + count)
is VALID only if each control state's transitions are CONTIGUOUS in the array.
The RTN build pushes transitions in production order (the q_in of with multiple
productions are scattered), which breaks the contiguity. The fix: sort the
transitions by (q, a, top) at construction (the compile + the new). The sort is
a reordering (the transition set is unchanged), so the machine's language, the
determinism, and the bitvec identity chain are all preserved (the three-way
identity is the same records, the same order across every tier). Theorem: after the sort, for
every q, the range [ctrl_offsets[q], +ctrl_counts[q]) contains exactly q's
transitions. Proof: the sort groups by q (the primary key), so q's records are
contiguous; the first-occurrence is the range start and the count is the range
length. QED. The oracle is the direct range check (the proof_csr_is_valid).

## The geometry (the conceptual primitives, the pictograms)

The notation key (the bespoke, the no wordsmithing):
- `o` = a control node (the q)
- `|` = the column (the stack, the top is up)
- `->` = a terminal shift (the pass-through, the column is unchanged)
- `v` = a call (the push, the column grows)
- `^` = a return (the pop, the column shrinks)
- `<` = a choice (the branch, the column is unchanged, the node splits)
- `[ ]` = a partition (the disjoint + the total)
- `x` = an overlap (the two terminals share an input)
- `D()` = the displacement (the path relation on the config space)

**1. Config space = Q x Gamma* (the node the column):**
```
            o q_(S,1)
            |
            | ret_A
            |
            Z0
   = a config (q, col)  a point in the product space
```

**2. The 5 RTN micro-ops (the geometry of delta):**
```
   shift a          call eps            return eps           choice eps
   o -> o'          o v o_in            o ^ o'               o < o1
   |                | ret               | ret'               |
   Z0               Z0                  Z0                   Z0
   (col =)          (col push)          (col pop)            (col =, branch)
   < the pass-through (the no control edge) >
```

**3. M(c) = the allowed-input set at a config (the settled gate):**
```
   o q_(S,1)
   |
   | ret_A
   Z0
   M(c) = { a : (q, a, top) -> (q', push) }  =  the allowed inputs here
   = the CSR row (the O(1-3) scan)
```

**4. B : Sigma -> 2^V (the terminal -> the map): a PARTITION iff a FUNCTION:**
```
   V (the input vocabulary)
   +---------------------------------------------+
   | B(s1)     B(s2)      B(s3)                 |
   | {a,b}     {c,d,...}  {}                    |
   +---------------------------------------------+
   disjoint+total  =>  B^-1: V -> Sigma is a FUNCTION (the DPDA)
   overlap         =>  B^-1: V x Sigma is a RELATION (the NPDA, the frontier)
```

**5. D(t) = the PATH RELATION on the config space (the CFGzip Theorem 2):**
```
   t = (a1...an)  =>  D(t) = { (c_in, c_out) : c_out reachable from c_in via t }

   c_in        c_out
   o q1        o q'
   | ret       | ret'
   Z0          Z0
   <---- path t ---->
   = the set of (in_config, out_config) pairs (the pure, the context-free)
```

**6. The bridge as a dimensional relation (the overlap + the exclusion):**
```
   terminals (rows) x inputs (cols)  =  the bipartite bridge
   s1  +--+--+--+--+
       |a |b |  |c |     a,b in s1 AND c in s2; d in NONE (the exclusion)
   s2  +--+--+--+--+
       |  |  |c |d |     d in s2 (the overlap if d also in s1)
   s3  +--+--+--+--+
       |  |  |  |  |
   inputs: a  b  c  d
   = the bridge is a RELATION (the 2D), not just a function (the 1D)
```

## The proofs (the test suite)

The test suite (tests/pda_tests.rs) proves:
- The language correctness (the {a^n b^n} accepts/rejects).
- The determinism (the discriminating, the no-dup-key for the deterministic + the
  dup-key for the ambiguous).
- The bounded stack (the reachable depth <= D, the per-step push bound).
- The mask fidelity (the PSC classifier == the machine's legal inputs over the
  full config space).
- The projection (the project(K) == the K sequential steps, the break on
  divergence).
- The state provenance (the RTN state -> nonterminal projection, the inclusive
  + exclusive oracle, the phase-homology).
- The epsilon-closure advance (the step_batch / the project_batch follow the
  epsilon moves to the terminal state, the no stuck call dots).
- The mask/advance consistency (the mask_at_cfg == the advance_eps-able inputs
  over the reachable config space, the proof_mask_batch_consistent_with_advance_eps).
- The bitvec round-trip (the lossless serialization, the exact POD size).
- The SWYB soundness (the d_H <= H for all reachable, the d_H = 0 unconditional
  for the accepting).
- The PSC codebook (the config -> the mask-class -> the VOB).
- The token spanner (the T_inv).
- The batched invariants (the batched == the scalar per-item).
- The CUDA graph (the DAG + the mock replay).
- The no-unsafe (the source-scan).
- The differential vs the independent oracle (the CFG membership, the {a^n b^n}
   + the balanced-parens + the multi-nonterminal, the exhaustive + the boundary
   corpora, the exact acceptance count).
- The CSR validity (the sorted-by-q array makes the first-occurrence + count range
   EXACT: every record in [offsets[q], +counts[q]) has t.q == q, and the count
   equals q's total; the proof_csr_is_valid, the scattered q_in regression).
- The settled-mask CSR == the linear reference (the O(1-3) mask_at_cfg_settled
   equals the order-independent scan over the full (q, top) space; the
   proof_mask_settled_csr_equals_linear).
- The epsilon-closure mask CSR == the linear reference (the mask_at_cfg equals
   the order-independent closure over the reachable config space; the
   proof_mask_at_cfg_csr_equals_linear).
- The pass-through soundness (the is_passthrough == the identity-stack input-
   consuming record exists, the inclusive + exclusive; the
   proof_is_passthrough_sound).
- The run exactness (the passthrough_run == the consecutive terminal shifts, the
   bounded-by-production; the proof_passthrough_run_exact).
- The scattered-choice regression (the advance_eps + the mask handle the q_in
    with multiple productions, the language still matches the CFG oracle; the
    proof_advance_eps_scattered_choice).

## The CYK acceptance (the CFG word problem)

The NPDA acceptance (the accepts_npda) is the CFG word problem (the L(M) ==
L(G), the RTN compilation preserves the language). It is decided by the CYK
algorithm (the O(n^3) decidable word problem for context-free languages, the
Hopcroft-Ullman), NOT by the PDA frontier simulation (the exponential, the
unbounded stack). The CYK builds the table V[i][j] = the set of nonterminals A
with A =>* w[i..i+j]. The acceptance is S in V[0][n].

The CYK is implemented k-ary (the no the binary-only stub): every production
A -> [s1..sm] (the m >= 0) is handled by a partition DP (the split of
w[i..i+j] into m consecutive, possibly-empty, spans, one per symbol). The
fixpoint over the productions resolves the unit + the chain dependencies (the
A -> B -> C). This is the verbatim transcription of the rhs_closure from the
pczarn/cfg crate (the fixpoint over the RHS properties, the transitive
closure).

The n=0 (the empty input) is handled by the derives_eps (the epsilon closure
of the start, the no the CYK table).

CITATIONS:
- Hopcroft, Motwani, Ullman, "Introduction to Automata Theory, Languages, and
  Computation" (the CYK algorithm, the O(n^3) CFL word problem).
- pczarn/cfg (the rhs_closure, the fixpoint over the productions, the
  transitive closure of the RHS property).
- CFGzip (Sullivan & Koller, arXiv:2605.29986) the GNF PDA (the single-state,
  the stack of nonterminals, the delta_G(a,A) = {beta : A -> a beta}).

## The Earley per-step mask (the O(n^3) chart)

The per-step mask (the allowed inputs at position i) is computed via the
Earley chart (the O(n^3) for ambiguous grammars, the O(n) for LR(k) with the
Leo 1991 optimization). The Earley state sets S[0..=n] where S[i] contains all
Earley items (production, dot, start) representing valid partial parses at
position i. The per-step mask at position i is the set of terminals a such that
there exists an item (p, dot, start) in S[i] with rhs[dot] == a (the terminal
at the dot).

The three Earley operations (the Hopcroft-Ullman):
- Predict: for (p, dot, start) where rhs[dot] is a nonterminal Y, add
  (q, 0, dot) for each production q of Y to the current state set.
- Scan: for (p, dot, start) where rhs[dot] is a terminal a, if a == w[i],
  add (p, dot+1, start) to S[i+1].
- Complete: for (p, dot, start) where dot == rhs.len() (complete), find all
  items (q, dot2, start2) in S[start] where rhs2[dot2] == lhs_p, and add
  (q, dot2+1, start2) to the current state set.

The earley_per_step_mask replaces the exponential advance_eps_set for the
per-step mask (the Qwen nested one_or_more loops, the frontier grows as
num_states * num_stack_syms^k after k steps). The advance_eps_set is retained
for the per-step frontier (the step_batch), but the Qwen test uses the Earley
mask (the polynomial).

CITATIONS:
- Earley (1970) "Efficient Parsing of Context-Free Grammars" (the O(n^3) chart
  parsing).
- Leo (1991) "A Practical Algorithm for Primary Parsing" (the O(n) for LR(k)
  grammars, the Aycock-Horspool optimization).
- Sun et al. (arXiv:2506.01151) "Earley-Driven Dynamic Pruning for Efficient
  Structured Decoding" (the ZapFormat, the dynamic pruning of dead Earley
  states, the state cache).
- pczarn/cfg (the Cfg::column, the DotInfo, the Earley column).

## The token spanner (the T_inv, the GreatGramma, the approximation)

The token spanner (the spanner.rs) maps (the lexer state, the terminal
sequence) to the tokens that produce that sequence from that state (the
GreatGramma T_inv, the arXiv:2502.05111 Def 3.4). The exact GreatGramma T_inv
is:

    T_inv(q, T1...TkT) = {t : q ->^{t:T1...Tk} q' in delta and T in Prod(q')}

where Prod(q') is the set of terminals PRODUCIBLE from q' (the the reachability
to accepting states, the Floyd 1962). This implementation uses the
LexerDfa::accepting (the the terminals that COMPLETE at the state) as a proxy
for the Prod (the the terminals producible from the state). The accepting is a
SUBSET of the Prod (the the terminals that complete immediately, the no the ones
that require further transitions). For lexers where the accepting set equals
the Prod set (the the terminal completes at the state it is produced), this is
exact. For lexers where the Prod extends beyond the accepting (the the terminal
can be produced by further transitions), this is an under-approximation (the
the T_inv misses some tokens). The LexerDfa trait would need a prod(q) method
(the the reachability) for the exact GreatGramma T_inv.

CITATIONS:
- GreatGramma (Park, Zhou, D'Antoni, arXiv:2502.05111) Def 3.4 (the T_inv,
  the Prod(q') reachability).
- Floyd (1962) (the the reachability algorithm, the the(q') computation).
- The LexerDfa::accepting (the the completed terminals, the no the producible).

## The max_closure_width (the measurement, the no the static cap)

The max_closure_width measures the maximum epsilon-closure width (the distinct
(state, top) pairs reachable via epsilon moves from any (q, top)). This sizes
the GPU frontier (the MAXF). The closure is over the (state, top) domain (the
num_states * num_stack_syms pairs), so the width is bounded by the domain size
(the no the overflow). The max_closure_width computes the EXACT maximum (the no
an underestimate) via the local BFS mirror (the mask_at_cfg returns the allowed
inputs, not the width). The proof_max_closure_width_is_exact test verifies:
the width <= the domain size (the exclusive), the width == the independent BFS
(the inclusive, the R3 different register).

CITATIONS:
- The epsilon closure over the (state, top) domain (the num_states * the
  num_stack_syms, the finite domain, the Hopcroft-Ullman PDA configuration
  semantics).
- The GPU frontier sizing (the MAXF, the device.md three-way layout-identity).

## The adversarial review (the proof status of every primitive)

Each primitive is classified as PROVEN (the inclusive + exclusive, the
independent oracle) or NOT-PROVEN (the gap). The MATHS grounding is from the
external sources (the no the repo's own comments).

PROVEN (the inclusive + exclusive, the independent oracle):
- advance_eps (the single-config): the deterministic case, the cycle_dpda test,
  the advance_eps_inclusion/exclusion_matches_reference_on_cycle.
- mask_at_cfg (the epsilon-closure mask): the proof_mask_at_cfg_csr_equals_
  linear, the GreatGramma Prop 3.5 stack invariance makes the top-dedup sound.
- mask_at_cfg_settled (the settled mask): the proof_mask_at_cfg_settled_is_
  precise, the inclusive + exclusive at every (q, top).
- displacement (the CFGzip relation): the proof_displacement_equals_reference,
  the monoid laws, the congruence.
- passthrough_run / is_passthrough (the linear-run): the proof_passthrough_run_
  exact, the proof_is_passthrough_sound, the independent linear chain-follow.
- accepts_via_eps (the epsilon-closure acceptance): the proof_accepts_via_eps_
  equals_reference, the independent linear-scan reference.
- accepts_npda (the CYK): the differential tests, the a^n b^n, the balanced-
  parens, the multi-NT, the independent cfg_accepts oracle.
- accepts_dpda (the single-path): the a^n b^n, the scaling_small_to_large_
  inputs, the Sénizergues 1997 result (the DPDA language equivalence is
  decidable, the O(n^3) algorithm).

NOT-PROVEN (the gaps):
- accepts_dpda for GENERAL deterministic PDAs (the no just the a^n b^n). The
  Sénizergues 1997 result proves the equivalence is decidable, but the repo's
  accepts_dpda is the single-path simulation (the no the equivalence
  algorithm). The single-path simulation is correct for the deterministic case
  (the unique path), but it's not PROVEN against an independent oracle for
  general DPDA.
- summary.rs d_H for the UNBOUNDED stack (the no just the bounded H). The
  current BoundedSummary::compute(machine, h) takes an explicit h (the caller-
  supplied bound). The d_H is the (min,+) semiring fixed point (the Reps 2003,
  the "min-plus semiring finds the shortest trace"). The repo's reverse BFS is
  the correct mechanism, but it's bounded by the caller's h (the no the full
  unbounded d_H).
- max_closure_width (the measurement): this is a MEASUREMENT (the no a proof).
  It measures the domain (the distinct (state, top) pairs reachable). The
  correctness is on the mask_at_cfg closure (the proven). The max_closure_width
  itself is not proven (it's a measurement, the no a theorem).

MATHS GROUNDING (the external sources):
- Sénizergues 1997 (the DPDA language equivalence is decidable, the O(n^3)
  algorithm, the ICALP 1997).
- Reps, Schwoon, Jha, Melski 2003 (the weighted pushdown systems, the (min,+)
  dioid, the "min-plus semiring finds the shortest trace", the SAS 2003).
- GreatGramma (Park, Zhou, D'Antoni, arXiv:2502.05111) Prop 3.5 (the stack
  invariance, the acceptance invariant under the stack extension).
- CFGzip (Sullivan & Koller, arXiv:2605.29986) Theorem 2 (the displacement
  equivalence refines the syntactic congruence, the lossless compression).
- Hopcroft, Motwani, Ullman (the CYK algorithm, the O(n^3) CFL word problem).
- pczarn/cfg (the rhs_closure, the fixpoint over the productions, the
  transitive closure of the RHS property).