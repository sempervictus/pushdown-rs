# The research (the cited sources)

The published works this crate implements. The algorithms are proven results
from these sources; this crate is a faithful implementation, not an invention.

## The automata theory (the classical)

- A. Hopcroft, R. Motwani, J. Ullman, "Introduction to Automata Theory,
  Languages, and Computation," 3rd ed., Addison-Wesley, 2006. The PDA/DPDA/NPDA
  definitions, the acceptance modes, the Chomsky hierarchy, the CFG<->PDA
  equivalence.
- M. Sipser, "Introduction to the Theory of Computation." The DPDA determinism
  condition, the DCFL = the DPDA-recognizable languages.
- The Chomsky-Schutzenberger theorem: the CFLs = the NPDA-recognizable
  languages.

## The RISC micro-machine precedent (the IBM 801)

- J. Cocke, V. Markstein, "The evolution of RISC technology at IBM," IBM J.
  Res. Dev. 34(1), 1990. The RISC as the micro-machine that removes the
  microcode (the CISC interpreter) layer.
- The IBM 801, deployed as a vertical-microcode execution unit inside the IBM
  9370 mainframe: a RISC machine interpreting a CISC instruction stream.
- N. Shchibuk, the research graduate whom Cocke hired to implement the RISC VM
  atop the CISC in hand-assembler, the work that secured IBM funding for RISC.
  The precedent for "a small orthogonal micro-machine executing a complex
  instruction set" - exactly the CFG (the CISC) -> the DPDA (the RISC)
  relationship this crate implements.

## The grammar-constrained decoding (the arXiv)

- F. Alpay, B. Senturk, "Attention Meets Reachability: Structural Equivalence
  and Efficiency in Grammar-Constrained LLM Decoding," arXiv:2603.05540 (2026).
  The RTN compilation (Definition 5), the kappa(G) state count (Definition 10,
  Lemma 2), the SAC (Theorem 2).
- K. Park, T. Zhou, D. D'Antoni, "Flexible and Efficient Grammar-Constrained
  Decoding" (GreatGramma), arXiv:2502.05111 (2025). The token spanner (the
  T_inv), the stack invariance (Prop 3.5), the online mask (Alg 6).
- V. Collura et al., "Stay Within Your Bounds: Distance-Guided Decoding for
  Guaranteed Context-Free Grammar Compliance" (SWYB), arXiv:2608.28229 (2026).
  The bounded pushdown summary (the S_H), the on-the-fly config database, the
  tokenizer-aware consumption.
- Y. Li et al., "Efficient Grammar-Constrained Decoding via Parser Stack
  Classification" (PSC), arXiv:2608.03065 (2026). The parser-stack
  classification (the config -> the mask-class -> the VOB codebook).
- J. Chen et al., "Pre^3: Enabling Deterministic Pushdown Automata for Faster
  Structured LLM Generation," arXiv:2506.03887 (2025). The LR(1)->DPDA + the
  prefix-conditioned edges.

## The reachability + the weighted pushdown (the S_H foundations)

- The Bouajjani et al. (1997), the reachability for pushdown systems (the
  finite stack summaries + the saturation).
- The Reps et al. (2003), the weighted pushdown systems (the path
  quantities).

These underlie the SWYB bounded summary (the summary.rs).