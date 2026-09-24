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
accepts_dpda runs the single path; the accepts_npda runs the bounded BFS over
all paths.

## The bounded stack

For a grammar with max production length L, the stack depth is bounded by D = L
+ 1 (the pending nesting). The DPDA is GPU-resident because D is a small
constant for the target grammars (the JSON, the tool envelopes, the network
headers).

## The epsilon-closure advance

The PDA's advance by a terminal must resolve the epsilon moves first (the RTN
choice/call/exit/return are epsilon). The advance_eps(q, stk, a) follows the
epsilon closure from (q, stk) (the BFS over the epsilon moves, the bounded by
the MAX_EPS_CLOSURE cap) to the configs where the terminal move a is available,
then does the terminal move. This is consistent with the mask (the mask_at_cfg
is the epsilon-closure of the allowed inputs): the advance reaches exactly the
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
associative, and the PDA's transition function is deterministic (the single
path). QED. The oracle is the independent direct computation (the
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