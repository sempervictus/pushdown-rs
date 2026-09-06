# pushdown-rs documentation

The coherent doc set for the pushdown-rs core library.

## Start here
- [library.md](library.md) - the traits, the data structures, the
  implementations (the PdaMachine, the Pda trait hierarchy, the
  PdaStream, the service).
- [mathematics.md](mathematics.md) - the formal maths (the 7-tuple, the
  RTN compilation, the kappa(G), the proofs).

## The PDA variants
- [pda_variants.md](pda_variants.md) - the 9 variants (the NPDA, the DPDA,
  the epsilon, the final-state, the empty-stack, the VPA, the alternating, the
  one-way, the nested) - structure, utility, and implementation.

## The device (the SIMD + the CUDA)
- [device.md](device.md) - the device plan (the bitvec upload, the
  kernels, the on-device construction, the FFI, the CUDA graph, the
  three-way layout-identity proof).

## The integration (the downstream)
- [integration_plan.md](integration_plan.md) - the llguidance -> the
  pushdown-rs -> the attention-rs -> the xinfer flow.
- [finite_stateless_grammar.md](finite_stateless_grammar.md) - the six
  constrained properties that make a grammar compilable to a finite DPDA.
- [risc_micro_machine.md](risc_micro_machine.md) - the RISC micro-machine
  framing (the CFG as the CISC, the DPDA as the RISC, the IBM 801
  precedent).

## The research (the citations)
- [research.md](research.md) - the supporting research (the arXiv
  papers, the classical automata, the RISC precedent, the
  credits).

## The QA (the accuracy + the benchmarks)
- [qa.md](qa.md) - the battery of accuracy tests (the independent oracle,
  the native matcher, the nom/deku) + the benchmark results (the
  PDA speedup, the SIMD speedup).