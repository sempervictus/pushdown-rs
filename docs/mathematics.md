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

## The state provenance

The RTN state -> nonterminal projection (the state_provenance, the Definition 5):
for each control state, the nonterminal index it belongs to (the q_start -> S,
the q_A^in/q_A^out -> A, the dot(p,i) -> lhs(p)). It is a total function of -> N
of length kappa(G). The phase-homology invariant: the label is invariant along
a terminal run inside one production, and changes exactly at the call/return
boundaries (the callee/caller non The consumer maps the nonterminal index to a
semantic region via the grammar's named nonterminals.

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