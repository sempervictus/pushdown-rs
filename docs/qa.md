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

## The test suite (the 30 tests + the 1 doc-test)

- The {a^n b^n} DPDA: the accepts ab/aabb/aaabbb, the rejects aab/abab.
- The RTN compilation: the kappa(G) exact, the deterministic + the
  ambiguous cases.
- The bitvec: the round-trip lossless, the truncation rejected.
- The scaling: the small-to-large inputs.
- The proofs: the determinism, the bounded stack, the mask fidelity, the
  projection, the SWYB soundness, the PSC codebook, the token spanner, the batched
  invariants, the CUDA graph, the no-unsafe.
- The differential: the PDA == the independent oracle (the {a^n b^n} + the
  balanced-parens).
- The non-deterministic regex: the [a-z]+ uses the accepts_npda.

Run: `cargo test` (the 30 tests + the 1 doc-test, the all green).