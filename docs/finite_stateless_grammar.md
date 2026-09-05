# The Stateless Finite Grammar (the constrained properties)

A stateless finite grammar is a CFG G = (N, Sigma, P, S) that satisfies the
constrained properties below. These properties are what make G compilable to a
finite, uploadable, SIMD-executable DPDA table.

## The six constrained properties

### 1. Finite
|N|, |Sigma|, |P| are all finite. The grammar is a finite object, not a
procedure. Consequence: the derived automaton has a finite state space.

### 2. Stateless
There is no explicit state machine in the grammar. The states are DERIVED (the
LR(1) item-sets, or the RTN control states q_(p,i)). The grammar is the source;
the automaton is the product. Consequence: the state space is a function of the
grammar structure, not of an independent state definition.

### 3. Bounded control
The derived control-state count is EXACTLY:

    kappa(G) = 1 + 2|N| + sum_{p in P} (|rhs(p)| + 1)

(the Lemma 2 of arXiv:2603.05540). The 1 is the q_start; the 2|N| is the
q_A^in + q_A^out per nonterminal; the sum is the dot-position chain per
production. Consequence: the control space is finite AND computable from the
grammar alone (the no simulation needed).

### 4. Deterministic (DCFL)
If G has no reduce/reduce conflict (the LR(1) property), the RTN compilation is
a DPDA. The DCFLs are exactly the DPDA-recognizable languages (the Valiant
1973 result). Consequence: the deterministic simulation is a SINGLE PATH (the
no BFS blowup), the SIMD-friendly.

### 5. Bounded pushdown
The stack depth is bounded by D = max_{p in P} |rhs(p)| + 1. A reduce of a
production of length n pops n and pushes 1 (the net n-1); the maximum pending
nesting is the longest production. Consequence: the stack is a FIXED-SIZE array
(the GPU-resident, the no unbounded allocation).

### 6. Finite token spanner
The (q_lex, terminal-sequence) -> tokens mapping (the T_inv, the
arXiv:2502.05111) is finite: the vocab x the DFA states x the terminal sequences.
Consequence: the token-level bridge is a PRECOMPUTED table (the source data for
the GPU construction, the no live DFA driving at mask time).

## Why these six together enable the GPU DPDA

The six properties jointly guarantee:
- The state space is finite (1, 3) and small (the kappa).
- The simulation is deterministic (4) - the single path, the SIMD.
- The stack is bounded (5) - the fixed-size array, the GPU-resident.
- The token bridge is precomputed (6) - the table lookup, the no live DFA.

A grammar that violates any of the six (the unbounded stack, the non-deterministic
choice, the infinite state space) CANNOT be compiled to a finite GPU DPDA - it
must stay on the CPU (the general NPDA, the Earley parser).

## The implementation in this crate

- The finite + the stateless: the Grammar trait (compile.rs) - the N, the Sigma,
  the P, the S are finite + the states are derived (the RTN compilation).
- The bounded control: the kappa function (compile.rs) - the exact state count.
- The deterministic: the is_deterministic (machine.rs) - the at-most-one
  transition check.
- The bounded pushdown: the max_stack bound (machine.rs) - the fixed-size
  stack array.
- The finite token spanner: the TokenSpanner (spanner.rs) - the precomputed
  T_inv table.