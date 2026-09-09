# The pushdown-rs core library

The traits, the data structures, the implementations.

## The trait hierarchy (the pda.rs)

The common Pda trait is the 7-tuple (Q, Sigma, Gamma, delta, q0, Z0, F):

```
   Pda (the 7-tuple)
   |--  the states(), the inputs(), the stack_syms(), the start_state(),
   |  the start_stack(), the accepting(), the transition(q, a, top)
   |
   +-- Npda          the accepts_npda(w, max_stack, max_configs) - the BFS
   +-- Dpda          the is_deterministic() + the accepts_dpda(w) - the single path
   +-- EpsilonPda    the marker (the machine has the epsilon transitions)
   +-- FinalStatePda the marker (the acceptance by the F)
   +-- EmptyStackPda the marker (the acceptance by the empty stack)
   +-- VisiblyPushdown the is_call/is_return/is_internal (the VPA)
   +-- AlternatingPda  the marker (the AND/OR branching)
   +-- OneWayStack     the marker (the stack head moves only down)
   +-- NestedStack     the marker (the stack holds sub-stacks)
   +-- PdaStream       the step_batch/mask_batch/project_batch (the batched pipeline node)
```

The PdaStream is the primary batched interface (the batched pipeline node): the host sends
a batch of configs, the device returns a batch of results. The scalar
accepts_dpda/accepts_npda are the reference (the correctness oracle).

## The concrete machine (the machine.rs)

The PdaMachine is the concrete 7-tuple with u32 IDs:

```
   PdaMachine {
     num_states, num_inputs, num_stack_syms,
     transitions: Vec<Transition>,   // the (q, a, top) -> (next_q, push)
     accepting: Vec<u32>,           // the F
     start_state, start_stack,      // the q0, the Z0
     state_provenance: Option<Vec<u32>>,  // the RTN state -> nonterminal projection
   }
```

The Transition is the (q, a, top) -> (next_q, push). The a is num_inputs for
the epsilon. The push is the stack string that replaces top (empty = pop).

The state_provenance is the RTN state -> nonterminal projection (the Definition 5
of arXiv:2603.05540): for each control state, the nonterminal index it belongs to
(the q_start -> S, the q_A^in/q_A^out -> A, the dot(p,i) -> lhs(p)). It is the
generic primitive for grammar-phase / region awareness (the consumer maps the
nonterminal index to a semantic region via the grammar's named nonterminals).
Some for RTN-compiled machines, None for hand-built ones.

The key groups:
- The lookup(q, a, top) - the linear-scan reference.
- The build_index() + the lookup_indexed - the O(1) hash index.
- The accepts / the accepts_sized - the universal (the auto-selects the DPDA
  vs the NPDA by the is_deterministic).
- The accepts_dpda / the accepts_npda - the single-path / the BFS.
- The mask_bits(state) - the O(num_inputs) mask-input scan.
- The provenance_of(q) - the RTN state -> nonterminal projection (the O(1) lookup).
- The advance_eps(q, stk, a) - the epsilon-closure advance (the BFS over the
  epsilon moves to the terminal state, then the terminal move; the no stuck
  call dots). The step_batch / the project_batch use it.
- The step_batch / the mask_batch / the project_batch - the PdaStream (the
  batched pipeline node).
- The step_batch_into / the step_batch_simd / the project_batch_simd - the
  no-alloc + the SIMD variants.
- The to_bitvec / the from_bitvec - the lossless POD encoding (the state_provenance
  is the trailing optional suffix).

## The compilation (the compile.rs)

The Grammar trait abstracts the CFG (the N, the Sigma, the P, the S). The
compile(g) function does the RTN (recursive-transition-network) compilation:

```
   the control states Q = {q_start} U {q_A^in, q_A^out : A in N}
                       U {q_(p,i) : p in P, i in 0..=|rhs(p)|}
   the stack alphabet Gamma = {bot} U {q_(p,i) : p in P, i}  (the return addresses)
```

The transitions are the start, the choice, the terminal, the call, the return,
the exit (the Definition 5 of arXiv:2603.05540). The exact control-state
count is kappa(G) = 1 + 2|N| + sum_p(|rhs(p)|+1) (the Definition 10 + the
Lemma 2).

The Cfg is the concrete CFG with u32 symbol IDs (the nonterminals are
0..num_nonterminals, the terminals are num_nonterminals..(num_nonterminals
+ num_terminals)). The validate() proves the input is a valid CFG (the
lhs is a nonterminal, the rhs symbols are defined, the start is a
nonterminal).

## The bitvec (the bitvec.rs)

The to_bitvec serializes the machine to a flat BitVec (the POD, the no
pointers, the no Rust types). The from_bitvec deserializes + re-validates.
The layout is the header (the 7 u32s) + the accepting + the transitions (the
q, a, top, next_q, push_len, push[]). This is the GPU-uploadable payload.

## The device tiers (the simd.rs, the service.rs, the graph.rs, the cuda.rs)

- The simd.rs - the rten-simd vectorized ops (the MaskOp broadcast, the
  StepBatchOp). The simd feature (default on).
- The service.rs - the PdaService (the packet-in/packet-out device
  model, the batched pipeline node boundary).
- The graph.rs - the PdaGraph (the CUDA DAG of the kernel nodes, the
  CUDA graph host-side interface, the mock_replay).
- The cuda.rs - the CudaPackage (the bitvec + the primitives, the
  H2D payload) + the FFI declarations (the cuda feature).

The three-way layout-identity invariant: the scalar == the SIMD batch == the
CUDA kernel, all consuming the same bitvec. This is the proof that the device
port is correct without a GPU in the loop.

## The visualization (the viz.rs)

The viz.rs is a pure diagnostic VIEW over the machine (the no execution tier,
the no device-identity link). It renders the control-state graph (the states =
the nodes, the transitions = the labeled edges) as a human-meaningful picture:

- The to_svg(m, state_names, term_names) - a self-contained SVG (the zero deps,
  the no-unsafe). The layout is a left-to-right column flow (the BFS depth from
  the start). The start state is ringed green, the accepting set is filled, the
  epsilon moves are dashed + red, the terminal moves are solid + grey, and each
  edge is labeled with the input + the stack op (the pop / the keep / the push(k)).
- The to_dot(m, state_names, term_names) - a Graphviz DOT file (the run
  `dot -Tsvg` for a prettier auto-layout).
- The write_svg / the write_dot - the file writers.
- The dump_g(g, m, term_names, dir) - the ready-to-render helper: it pulls the
  state names from the rtn_state_names (the compile.rs) and writes both the
  .svg + the .dot to the dir.

The rendering invariants (the pure function of the machine): one primary node
per control state (the node count == the num_states), one edge per transition
(the edge count == the transitions.len()), and deterministic output (the same
machine renders identically, the groups are sorted). The viz_dump example
demonstrates it (the {a^n b^n}, the Dyck, the hand-built DFA-embedded).