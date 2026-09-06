# The PDA variants (the consolidated)

The 9 PDA variants, each with the maths, the structure, the utility, the
function, + the implementation in this crate.

## The NPDA (the non-deterministic)
- The maths: the delta is a RELATION (the multiple next-moves per
  (q, a, top)). The accepts if ANY computation path accepts.
- The structure: the same 7-tuple, the delta is a set-valued function.
- The utility: the general CFG recognizer (the CFLs). The the RTN
  compilation of an ambiguous CFG yields an NPDA.
- The function: the accepts_npda(w, max_stack, max_configs) - the bounded over
  the (q, stack) configs, bounded by the max_stack + the max_configs.
- The implementation: the Npda trait + the PdaMachine::accepts_npda (the
  BFS, the per-step dedup, the top-level acceptance).

## The DPDA (the deterministic)
- The maths: the delta is a PARTIAL FUNCTION (the at most one next-move
  per (q, a, top)). The accepts if the unique path accepts.
- The structure: the same 7-tuple, the delta is a function.
- The utility: the DCFLs (the JSON, the tool envelopes, the
  network headers). The the bounded/SIMD target (the single path, the no
  branching).
- The function: the accepts_dpda(w) - the single-path simulation. The
  is_deterministic() - the at-most-one check.
- The implementation: the Dpda trait + the PdaMachine::accepts_dpda (the
  single path, the deterministic epsilon-closure).

## The EpsilonPda (the epsilon transitions)
- The maths: the delta includes the (q, epsilon, top) moves (the no input
  consumed).
- The utility: the RTN compilation uses the epsilon moves (the choice,
  the call, the return, the exit are all epsilon).
- The implementation: the EpsilonPda marker trait. The the transition(q, None,
  top) is the epsilon move.

## The FinalStatePda (the acceptance by the F)
- The maths: the accept iff the input is exhausted AND the control state is in
  the F.
- The utility: the RTN compilation sets the F = {q_S^out} (the exit of
  the start nonterminal).
- The implementation: the FinalStatePda marker trait. The the accepting field.

## The EmptyStackPda (the acceptance by the empty stack)
- The maths: the accept iff the input is exhausted AND the stack is empty.
- The utility: the equivalent to the final-state for the DPDA (the
  standard theorem).
- The implementation: the EmptyStackPda marker trait.

## The VisiblyPushdown (the VPA)
- The maths: the input alphabet is partitioned into the call / the return /
  the internal symbols. The push is forced by the call, the pop by the return,
  the no-op by the internal.
- The utility: the VPLs (the well-nested structures, the XML, the
  the JSON). The the better closure properties than the DCFLs.
- The implementation: the VisiblyPushdown trait (the is_call/is_return/
  is_internal).

## The AlternatingPda (the alternating)
- The maths: the transitions carry the AND/OR branching (the universal +
  the existential quantification over the next configs).
- The utility: the strictly more expressive than the NPDA (the
  context-sensitive languages).
- The implementation: the AlternatingPda marker trait.

## The OneWayStack (the one-way stack)
- The maths: the stack head moves only downward (the no re-reading of the
  deeper symbols).
- The utility: the restricted PDA (the a subset of the CFLs).
- The implementation: the OneWayStack marker trait.

## The NestedStack (the nested stack)
- The maths: the stack holds sub-stacks (the nested values).
- The utility: the strictly more powerful than the PDA (the indexed
  languages).
- The implementation: the NestedStack marker trait.

## The PdaStream (the batched / the batched pipeline node)
- The maths: the batch of configs is processed in parallel (the batched
  node model). The the step_batch / the mask_batch / the project_batch. The
  step_batch and the project_batch use the epsilon-closure advance (the
  advance_eps, the no stuck call dots) — consistent with the mask (the
  mask_at_cfg, the epsilon-closure of the allowed inputs).
- The utility: the SIMD/GPU execution (the B sequences in vectors).
- The implementation: the PdaStream trait + the PdaMachine's batched methods +
  the PdaService (the packet-in/packet-out).