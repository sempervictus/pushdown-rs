# pushdown-rs

Pushdown automata (PDA / DPDA / NPDA and variants) as composable traits,
compiled from any finite-state grammar, producing flat bitvec machine encodings
for CPU, SIMD, and GPU execution.

A **pushdown automaton** is the right7-tuple `(Q, Sigma, Gamma, delta, q0, Z0, F)`:
the finite control `Q`, the input alphabet `Sigma`, the stack alphabet `Gamma`,
the transition relation `delta`, the start state `q0`, the start stack symbol
`Z0`, the accepting states `F`. Unlike a DFA, the stack carries the nesting
memory that lets it recognize context-free (not just regular) structure - which
is what JSON, tool-call envelopes, and network headers require.

## Why this exists

Grammar-constrained LLM decoding needs a machine that (a) recognizes the
grammar's language exactly, (b) is small enough to live on a GPU, and (c) can
project K steps ahead for speculative drafting. A DFA cannot do (a) for
context-free grammars. A pushdown automaton can, and when the grammar is
deterministic (DCFL) the control state space is finite and the stack is bounded
by the longest production - exactly the shape a GPU table wants.

The framing is the RISC micro-machine. A context-free grammar is the CISC of
formal languages: a complex, high-level instruction set (the productions) that
a general parser (the Earley chart) interprets with unbounded state. The
pushdown automaton is the RISC micro-machine that executes it: a small
orthogonal set of primitive operations (shift, reduce, accept, call, return)
over a finite control and a bounded stack. This is the same relationship the
IBM 801 RISC had to the System/370 CISC: the 801 was deployed as a
vertical-microcode execution unit inside the 370 line, a simple machine
interpreting a complex instruction stream [1][2]. The RTN compilation in this
crate is the lowering pass that turns the grammar's productions into pushdown
micro-ops, exactly as a compiler lowers CISC to RISC.

[1] J. Cocke, V. Markstein, "The evolution of RISC technology at IBM,"
    IBM J. Res. Dev. 34(1), 1990.
[2] IBM 801; the 801 as a vertical-microcode execution unit in the IBM 9370.

## How it fits together (the newcomer's view)

```
   a grammar (any source)          this crate                    the device
   -------------------            ----------------              ----------------
   JSON schema        \
   Lark / BNF          >-- Cfg -- compile() --> PdaMachine -- to_bitvec() --> GPU
   network protocol     (the CFG)  (the RTN      (the 7-tuple)   (the POD      (the kernels
   a hand-built CFG     lowering)   states +          |          table)        read the
                                   transitions)      |
                                                    accepts() / step_batch() / project_batch()
                                                    (the CPU/SIMD execution)
```

1. You describe a grammar (the JSON schema, the protocol, the network protocol).
2. `compile()` lowers it to a PDA (the RTN construction, the exact kappa(G)
   states).
3. `to_bitvec()` serializes the PDA to a flat POD table (the GPU payload).
4. The PDA runs on CPU (the accepts, the step_batch) or on the GPU (the kernels
   read the bitvec).

The whole point: a context-free grammar becomes a small, bounded, GPU-resident
state machine that can validate + guide token generation at line rate.

## The modules

| Module | What it is |
|---|---|
| `pda` | The trait hierarchy: `Pda` (the 7-tuple), `Npda`, `Dpda`, `EpsilonPda`, `FinalStatePda`, `EmptyStackPda`, `VisiblyPushdown`, `AlternatingPda`, `OneWayStack`, `NestedStack`, `PdaStream` (the batched/batched interface) |
| `machine` | `PdaMachine` - the concrete 7-tuple with u32 IDs; the NPDA/DPDA simulations; the universal `accepts` (auto-selects deterministic vs not); the batched `step_batch`/`mask_batch`/`project_batch` |
| `compile` | The `Grammar` trait (any CFG source) + the RTN compilation to a PDA + the exact `kappa(G)` state count + `validate_cfg` |
| `bitvec` | The flat POD encoding (`to_bitvec`/`from_bitvec`) - the GPU-uploadable payload |
| `simd` | The rten-simd vectorized ops (`MaskOp` broadcast, `StepBatchOp`) - the `simd` feature (default on) |
| `summary` | The SWYB bounded pushdown summary (the reachability labels + the distance-to-acceptance `d_H`) |
| `mask_class` | The PSC mask-classification (the config -> mask-class -> VOB codebook) |
| `spanner` | The GreatGramma token spanner (the `(lexer_state, terminal_seq) -> tokens`) |
| `oracle` | An independent CFG membership oracle (the ground truth for differentials) |
| `service` | The `PdaService` - the packet-in/packet-out device model (the host sends a batch, the device returns a batch) |
| `cuda` | The `CudaPackage` (the bitvec + the source primitives) + the FFI declarations behind the `cuda` feature |

## Usage

```rust
use pushdown_rs::compile::{Cfg, Grammar};
use pushdown_rs::pda::Dpda;

// the {a^n b^n} grammar: S -> a S b | a
let g = Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![1])]);
let m = pushdown_rs::compile(&g).expect("compile");

assert!(m.is_deterministic());
assert!(m.accepts_dpda(&[0, 1]));        // "ab"
assert!(m.accepts_dpda(&[0, 0, 1, 1])); // "aabb"
assert!(!m.accepts_dpda(&[0, 0, 1]));    // "aab"
```

The universal `accepts` auto-selects the right simulation:

```rust
let ok = m.accepts(&input); // DPDA path if deterministic, NPDA path otherwise
```

## The correctness execution tiers

1. **Scalar** - `accepts_dpda`/`accepts_npda`/`mask_bits` (the reference, always available).
2. **SIMD** - the `PdaStream` batched ops + the rten-simd `MaskBroadcastOp`/`StepBatchOp` (the `simd` feature, default on). The host sends a batch of configs, the device returns a batch of results - the batched pipeline node model. The `MaskBroadcastOp` vectorizes the logit+mask add across the vocab (proven bit-exact vs scalar by `tests/simd_accuracy.rs`).
3. **CUDA** - the `CudaPackage` bitvec + the `ffi/pda_ffi.h` contract. The same bitvec the SIMD tier uses, consumed by GPU kernels (the layout-identity invariant: CUDA == SIMD == scalar).

## Correctness model

Every operation is proven against an **independent oracle** (`oracle::cfg_accepts`,
zero shared code with the PDA). The test suite (30+ tests) covers: the language
membership, the determinism, the bounded stack, the mask fidelity, the
projection-equals-sequential, the bitvec round-trip, the SWYB soundness, the PSC
codebook, the token spanner, the batch invariants, and the no-`unsafe` guarantee.

## Features

- `simd` (default) - the rten-simd vectorized ops.
- `cuda` - the FFI declarations (the `extern "C"` block; the implementations
  live in the consuming crate).

Build without SIMD: `cargo build --no-default-features`.

## Tests and Benches

- `cargo test` - 30/30 core proofs (language membership, determinism,
  bounded stack, mask fidelity, projection, bitvec round-trip, SWYB soundness,
  PSC codebook, token spanner, batch invariants, no-unsafe, differential vs
  independent oracle).
- `cargo test --features simd --test simd_accuracy` - 6/6 SIMD accuracy proofs
  (MaskBroadcastOp == scalar, all-allowed identity, all-disallowed -inf, u8
  mask dispatch, step batch, compute_bias patterns).
- `cargo bench --bench simd_bench` - Criterion: scalar vs SIMD at 256/4K/32K/248K
  vocab. At 4K vocab, SIMD is 2x faster (172ns vs 345ns); at 248K both are
  memory-bound (~33 GiB/s).

## References

- Cocke & Markstein, "The evolution of RISC technology at IBM," IBM J. Res.
  Dev. 34(1), 1990 - the RISC micro-machine precedent (the 801 as a
  vertical-microcode execution unit atop the System/370 CISC).
- IBM 801; the 801 as a vertical-microcode execution unit in the IBM 9370.
- Alpay & Senturk, "Attention Meets Reachability", arXiv:2603.05540 - the RTN
  compilation (Definition 5) + the kappa(G) state count (Definition 10, Lemma 2).
- Hopcroft, Motwani, Ullman, "Introduction to Automata Theory, Languages, and
  Computation" (3rd ed.) - the PDA/DPDA/NPDA definitions, the acceptance modes.
- Sipser, "Introduction to the Theory of Computation" - the DPDA determinism
  condition, the DCFL = the DPDA-recognizable languages.
- JFLAP (Rodger & Finley) - the {a^n b^n} DPDA tutorial (the push/pop construction).
- GreatGramma (Park et al., arXiv:2502.05111) - the token spanner (the T_inv) +
  the stack invariance (Prop 3.5) + the online mask (Alg 6).
- SWYB (Collura et al., arXiv:2608.28229) - the bounded pushdown summary (the
  S_H) + the tokenizer-aware consumption.
- PSC (Li et al., arXiv:2608.03065) - the parser-stack classification (the
  mask-classes).
- Pre3 (Chen et al., arXiv:2506.03887) - the LR(1)->DPDA + the prefix-conditioned
  edges.

## Credits

This work is credited to Semper Victus Engineering, with thanks to WhiteFiber
for supporting greenfield designs for safe code. Special thanks to Norman
Schibuk for doing this by hand in assembler code at IBMR under John Cocke to
enable IBM funding for RISC - we stand on the shoulders of giants.

The algorithms implemented here are proven results from the cited arXiv papers;
this crate is a faithful implementation of those proofs, not an invention.
