# AGENTS.md — pushdown-rs (the work-flow controller)

This file is a PDA over an agent's work-stream. It does not describe the
constraints; it ENFORCES them. The agent's natural token distribution degenerates
into the fault bits F1..F7. The XOR mask is the constraint that forbids each
degenerate continuation and requires its complement. A lesser model is steered
mechanically: it must emit the required tokens or its output does not parse and
the commit is rejected.

## The framing (why the agent works this way)

pushdown-rs is a RISC machine generator. The RTN compilation lowers a complex
CFG (the CISC spec) to a small orthogonal DPDA (the RISC micro-ops: shift,
reduce, accept, call, return) via six constrained properties, and the test
suite proves each property (inclusive + exclusive). The machine's CORRECTNESS is
established by the scalar -> SIMD -> GPU identity chain: each tier proves the
next WITHOUT the next tier's hardware (the scalar is the ground-truth oracle;
the SIMD mock is the GPU's oracle). The agent's work-stream is the same lowering
pass: decompose the task into provable primitives, prove each, compose the
construction, and preserve the identity chain. An agent that ships an unproven,
complex, tier-specific monolith is the degeneration (the CISC side). The XOR mask
below holds the agent in the RISC regime.

## The six properties -> the agent's invariants

Each property (docs/finite_stateless_grammar.md) is an invariant the agent's
work must satisfy:

| Property | Machine consequence | Agent invariant |
|---|---|---|
| Finite | finite state space | the task has a bounded scope, not an open-ended procedure |
| Stateless | states are derived from the grammar | the workflow states are derived from the task structure, not imposed |
| Bounded control | kappa(G) is exact | compute the exact bound, never a loose one (Gate 9) |
| Deterministic | single path, no BFS blowup | one correct continuation per phase (no option-spread) |
| Bounded pushdown | fixed-size stack (D) | the pending-work stack is bounded (no unbounded WIP accumulation) |
| Finite token spanner | precomputed table | mappings (primitive->proof, claim->citation) are precomputed, not live re-derived |

## The XOR mask (token-level gates; each cites the property it enforces)

The agent's natural distribution degenerates into the fault bits F1..F7. The mask
forbids each degenerate continuation and requires its complement.

- XOR(F1) confident closure -> falsify-first: the token "this is correct" is
  forbidden; a FALSIFIER ("false if <input>") must precede any correctness claim.
  (Deterministic: one refutation path, not a confidence spread.)
- XOR(F2) plausible fill -> provenance-required: an external claim with no source
  token is a PARSE ERROR; every claim carries (file:line) or (arXiv:xxxx Def N).
  (Finite token spanner: the claim->source map is precomputed, not invented.)
- XOR(F3) completion -> gap-mandatory: "all cases covered" is reachable only if a
  GAP: list is emitted (empty-and-stated, or non-empty). (Finite: the scope is
  bounded and named.)
- XOR(F4) simplification -> boundary-exhaustive: the set {0, 1, max, empty,
  just-in, just-out, out-of-range} must be present; a proof missing an edge token
  is rejected. (Bounded control: the edges are where kappa is exact.)
- XOR(F5) circularity -> register-shift: the oracle must be in a DIFFERENT
  register (a math definition / a 2nd impl / a combinatorial count), never the
  code under test. (Stateless: the reference is derived independently.)
- XOR(F6) premature "done" -> scratch-gate: volatile tokens go to a dot-file;
  "done" is reachable only from the production state (F1..F5 satisfied).
  (Bounded pushdown: the WIP stack is drained, not committed.)
- XOR(F7) assumption -> read-citation: every edit token must be preceded by a
  file:line read token; an edit without a read is a parse error.
  (Stateless: the state is derived from what was actually read.)

## The work-grammar (the lowering pass; COMMIT is the accepting state)

    READ --(read-citations)--> FRAME --(reuse-or-justify)--> HYPOTHESIS
         --(falsifier)--> ORACLE --(independent register)--> PROVE
         --(inclusive+exclusive, boundaries, GAP:)--> BENCH --(cost)-->
         DOC --(the academic record, the doc-sync invariant)-->
         SCRATCH --(volatile)--> PROMOTE --(production-gate)--> COMMIT

- READ: cite file:line for every existing behavior relied on. (Gate 0; XOR(F7).)
- FRAME: name the existing primitive to reuse, or justify a new one. (DRY; the
  RTN lowering -- compose, don't monolith.)
- HYPOTHESIS / ORACLE / PROVE / BENCH: the primitive is proven inclusive +
  exclusive and benched. (XOR(F1,F3,F4,F5).)
- DOC: update the doc that owns the change (the doc-sync invariant, the
  library/mathematics/pda_variants/qa/research/device mapping). The primitive
  and its proof travel with their doc entry.
- SCRATCH -> PROMOTE: volatile work in a dot-file; production code promoted only
  when proven. (XOR(F6).)
- COMMIT: the structured message below. (Accepting state.)

## The commit message (the acceptance test)

    <type>: <imperative summary>

    LOGIC + MATHS: <the theorem/property, why correct, the inclusive + exclusive
      argument, the decomposition (primitive -> construction), the six-property
      invariant(s) it upholds>

    CITATIONS:
      - <arXiv:xxxx Def N / file:line / docs/<mathematics|research|device|qa>.md §>

    CHECKLIST (all [x] or rejected):
      [ ] Falsifier before any correctness claim              [XOR(F1)]
      [ ] Every external claim sourced                        [XOR(F2)]
      [ ] GAP: list stated (empty+stated, or non-empty)       [XOR(F3)]
      [ ] Boundaries enumerated: 0,1,max,empty,just-in,just-out [XOR(F4)]
      [ ] Oracle in an independent register (not circular)   [XOR(F5)]
      [ ] Tracked tree commit-ready (volatile work was scratch) [XOR(F6)]
      [ ] Every edit preceded by a read-citation             [XOR(F7)]
[ ] DRY: reused existing primitives (the lowering, not a monolith)
       [ ] Doc-sync: the academic record updated (the library/mathematics/
/etc.
         entry owns this change, the doc-sync invariant)
       [ ] Device identity chain preserved (scalar==SIMD==GPU[==nostd])
       [ ] cargo test green

## The device-parity invariant (the correctness proof)

The RISC machine is CORRECT because it runs identically on scalar -> SIMD ->
,
and that identity IS the proof (docs/device.md §116-126, the three-way
layout-identity):

  1. scalar == the independent oracle (the language correctness, the ground truth)
  2. SIMD batch == scalar per-item (the batched invariant)
  3. GPU kernel == SIMD batch (the layout-identity, the SAME bitvec POD)

Each tier proves the next WITHOUT the next tier's hardware (the SIMD mock is the
GPU's oracle; the scalar is the SIMD's oracle). The machine "runs correctly on
scalar->SIMD->GPU and proves that correctness when they do" = the identity chain
holds.

nostd is the FOURTH link: it joins the chain (nostd == scalar, the
allocation-free path) and must pass the same identity tests.

AGENT RULE: any change to a tier or to the bitvec POD must PRESERVE the identity
chain. A tier-specific proof (correct on GPU but not shown == scalar) is a CISC
regression -- the correctness is established only by the chain, never by a single
tier in isolation.

## The doc-sync invariant (the academic record stays current)

The docs/ set is the crate's academic record (the theorems, the proofs, the
citations, the QA battery). A code change that is not reflected in the docs is an
INCOMPLETE change (the record diverges from the machine). Before COMMIT, update
the doc that owns each kind of change:

| Code change | Doc to update |
|---|---|
| A new/changed PdaMachine field or method (the API) | docs/library.md (the concrete machine + the key groups) |
| A new/changed theorem, invariant, or proof property | docs/mathematics.md (the formal maths + the proofs list) |
| A new/changed PDA variant or itsed-op semantic | docs/pda_variants.md (the 9 variants) |
| A new/changed test or bench (the QA battery) | docs/qa.md (the test suite + the benchmark results) |
| A new external result the crate now implements | docs/research.md (the citations) |
| A new constrained tier or the bitvec POD layout | docs/device.md (the three-way identity) |

The rule: the primitive and its proof travel together with their doc entry. If you
add a primitive (the state_provenance) you add its doc (the library.md API + the
mathematics.md theorem + the qa.md proof). A commit that adds code without the
matching doc entry is rejected (the record is stale).

## Commands

- Test: cargo test (all tiers), cargo test --features simd, cargo test --features cuda
- Bench: cargo bench
- no-unsafe proof: cargo test crate_is_unsafe_free
- device identity chain: cargo test proof_stream_step_batch_equals_scalar proof_simd_service_device_model