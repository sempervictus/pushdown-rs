# QA + the benchmarks (the accuracy + the speed)

The battery of accuracy + benchmark tests, with the results. This is the
evidence the crate is correct + fast.

## The accuracy battery (the 100% match)

### The independent oracle (the CFG membership)
The oracle::cfg_accepts is a naive recursive CFG membership test with ZERO
shared code with the PDA. The differential tests assert the PDA's accepts ==
the oracle's accepts over a corpus:
- The {a^n b^n} (the S -> a S b | eps): the 100% match.
- The balanced parentheses (the S -> ( S ) S | eps): the 100% match.

### The native matcher (the llguidance)
For the real grammars (the JSON, the regex, the xbot), the PDA's accepts ==
the llguidance native matcher's accepts, 106/106 inputs (the valid + the
invalid + the edge + the random corpus). This is the 100% gate.

### The proof/deku oracles
The nom parser combinators + the deku derive-struct parsers are independent
oracles. The PDA == the nom parser + the PDA == the deku parser on the BER/TLV,
the JSON, the regex cases.

## The benchmark results (the release build, the 248K vocab)

### The PDA vs the native compute_mask
| Grammar | PDA states | native compute_mask | PDA O(1) lookup | speedup |
|---|---|---|---|---|
| JSON-complex | 372 | 1.464 ms | 0.012 ms | ~120x |
| regex [a-z]+ | 7 | 1.086 ms | 0.007 ms | ~155x |
| xbot | 107 | 0.963 ms | 0.008 ms | ~120x |

The PDA's O(1) table lookup is ~120-155x faster than the native O(vocab) trie
walk.

### The SIMD vs the scalar
| Op | scalar | SIMD (rten-simd) | speedup |
|---|---|---|---|
| The mask broadcast | 443 ns | 5 ns | ~88x |
| The step_batch | 19.7 us | 730 ns | ~27x |

The SIMD win is real but modest at small batch (the dispatch overhead
dominates); it grows with the batch size (the batch model).

### The device (the PdaService, the packet-in/packet-out)
The device PDA (the step_batch) is slower than the scalar at small batch
(the per-item Vec allocation). The step_batch_into (the no-alloc) + the
step_batch_simd close this gap.

## The test suite (the 46 tests + the 1 doc-test)

- The {a^n b^n} DPDA: the accepts ab/aabb/aaabbb, the rejects aab/abab.
- The RTN compilation: the kappa(G) exact (the heterogeneous grammars), the
  deterministic + the ambiguous cases.
- The bitvec: the round-trip lossless, the truncation rejected, the exact POD
  size formula (the header + the accepting + the transitions + the provenance
  suffix).
- The scaling: the small-to-large inputs.
- The state provenance (the RTN state -> nonterminal projection, the
  Definition 5): the inclusive + exclusive oracle (the total + the disjoint
  families, the exact kappa length) across the edge-case grammars (the
  eps-production, the single nonterminal, the unit chain, the recursion), the
  phase-homology (the terminal/choice/exit/start moves preserve the phase,
  the call/return moves change it to the callee/caller), the bitvec round-trip.
- The proofs: the determinism (the discriminating, the no-dup-key vs the
  deterministic + the dup-key for the ambiguous), the bounded stack (the
  reachable depth <= D, the per push bound), the mask fidelity (the PSC
  classifier == the machine's legal inputs over the full config space), the
  projection (the project(K) == the K sequential steps, the break on
  divergence), the SWYB soundness (the d_H <= H for all reachable, the d_H = 0
  unconditional for the accepting), the PSC codebook, the token spanner, the
  batched invariants, the CUDA graph, the no-unsafe.
- The epsilon-closure advance (the step_batch / the project_batch follow the
  epsilon moves to the terminal state, the no stuck call dots): the S -> a A b,
  A -> c grammar advances through the A call + the differential oracle agrees.
- The differential: the PDA == the independent oracle (the {a^n b^n} + the
  balanced-parens + the multi-nonterminal grammar, the exhaustive + the
  boundary corpora, the exact acceptance count).
- The non-deterministic regex: the [a-z]+ uses the accepts_npda.

Run: `cargo test` (the 46 tests + the 1 doc-test, the all green).