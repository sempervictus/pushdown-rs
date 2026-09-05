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

## The proofs (the test suite)

The test suite (tests/pda_tests.rs) proves:
- The language correctness (the {a^n b^n} accepts/rejects).
- The determinism (the at-most-one check).
- The bounded stack (the push bound).
- The mask fidelity (the mask == the legal inputs).
- The projection (the project(K) == the K sequential steps).
- The bitvec round-trip (the lossless serialization).
- The SWYB soundness (the d_H upper-bound).
- The PSC codebook (the config -> the mask-class -> the VOB).
- The token spanner (the T_inv).
- The batched invariants (the batched == the scalar per-item).
- The CUDA graph (the DAG + the mock replay).
- The no-unsafe (the source-scan).
- The differential vs the independent oracle (the CFG membership).