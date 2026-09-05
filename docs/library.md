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
   }
```

The Transition is the (q, a, top) -> (next_q, push). The a is num_inputs for
the epsilon. The push is the stack string that replaces top (empty = pop).

The key groups:
- The lookup(q, a, top) - the linear-scan reference.
- The build_index() + the lookup_indexed - the O(1) hash index.
- The accepts / the accepts_sized - the universal (the auto-selects the DPDA
  vs the NPDA by the is_deterministic).
- The accepts_dpda / the accepts_npda - the single-path / the BFS.
- The mask_bits(state) - the O(num_inputs) mask-input scan.
- The step_batch / the mask_batch / the project_batch - the PdaStream (the
  the batched pipeline node).
- The step_batch_into / the step_batch_simd / the project_batch_simd - the
  no-alloc + the SIMD variants.
- The to_bitvec / the from_bitvec - the lossless POD encoding.

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
  the StepBatchOp). The the simd feature (default on).
- The service.rs - the PdaService (the packet-in/packet-out device
  model, the batched pipeline node boundary).
- The graph.rs - the PdaGraph (the CUDA DAG of the kernel nodes, the
  the CUDA graph host-side interface, the mock_replay).
- The cuda.rs - the CudaPackage (the bitvec + the primitives, the
  the H2D payload) + the FFI declarations (the cuda feature).

The three-way layout-identity invariant: the scalar == the SIMD batch == the
CUDA kernel, all consuming the same bitvec. This is the proof that the device
port is correct without a GPU in the loop.