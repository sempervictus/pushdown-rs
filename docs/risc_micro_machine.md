# The RISC micro-machine framing

The pushdown automaton is the RISC micro-machine for context-free grammars, in
the same sense the IBM 801 RISC was the micro-machine for the System/370 CISC.

## The analogy

| The CISC side | The RISC side |
|---|---|
| The CFG (the productions, the high-level "instructions") | The DPDA (the shift/reduce/accept micro-ops) |
| The Earley parser (the general interpreter, the unbounded chart) | The RN micro-machine (the finite control + the bounded stack) |
| The microcode (the CISC interpreter layer) | The RTN compilation (the lowering pass) |
| The compiler (the CISC -> the RISC) | The `compile()` (the CFG -> the PDA) |

The CFG is the "CISC instruction set": a complex, high-level specification that
a general interpreter (the Earley chart) executes with unbounded state. The DPDA
is the "RISC micro-machine": a small orthogonal set of primitives (shift,
reduce, accept, call, return) over a finite control + a bounded stack. The RTN
compilation is the "compiler" that lowers the CFG to the DPDA micro-ops.

## The precedent (the IBM 801)

Cocke and Markstein showed that removing the microcode layer (the CISC
interpreter) and exposing a small primitive set to the compiler removes the
per-operation overhead. The 801 was itself deployed as a vertical-microcode
execution unit inside the System/370 line: a RISC machine interpreting a CISC
instruction stream. Shchibuk framed the 801 as a virtual machine atop the CISC
to secure funding; the same framing applies here. The DPDA is a virtual machine
atop the CFG, and the RTN compilation is the lowering pass.

## The citations

- [1] J. Cocke, V. Markstein, "The evolution of RISC technology at IBM," IBM
  J. Res. Dev. 34(1), 1990.
- [2] IBM 801; the 801 as a vertical-microcode execution unit in the IBM 9370.
- [3] F. Alpay, B. Senturk, "Attention Meets Reachability: Structural
  Equivalence and Efficiency in Grammar-Constrained LLM Decoding,"
  arXiv:2603.05540 (2026), Definition 5 (the RTN compilation), Definition 10 +
  Lemma 2 (the kappa(G) state count).
- [4] K. Park, T. Zhou, D. D'Antoni, "Flexible and Efficient Grammar-Constrained
  Decoding" (GreatGramma), arXiv:2502.05111 (2025) - the token spanner (the
  T_inv), the stack invariance (Prop 3.5), the online mask (Alg 6).
- [5] V. Collura et al., "Stay Within Your Bounds: Distance-Guided Decoding for
  Guaranteed CFG Compliance" (SWYB), arXiv:2608.28229 (2026) - the bounded
  pushdown summary (the S_H), the on-the-fly config database.
- [6] Y. Li et al., "Efficient Grammar-Constrained Decoding via Parser Stack
  Classification" (PSC), arXiv:2608.03065 - the mask-classification (the
  codebook).
- [7] J. Chen et al., "Pre^3: Enabling Deterministic Pushdown Automata for
  Faster Structured LLM Generation," arXiv:2506.03887 (2025) - the LR(1)->DPDA
  + the prefix-conditioned edges.
- [8] A. Hopcroft, R. Motwani, J. Ullman, "Introduction to Automata Theory,
  Languages, and Computation," 3rd ed., Addison-Wesley, 2006.
- [9] M. Sipser, "Introduction to the Theory of Computation."