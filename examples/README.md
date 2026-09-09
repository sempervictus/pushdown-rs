# pushdown-rs examples

Runnable examples + the benchmarks. The `bench_realworld/` fixtures (the 248K
tokenizer + the grammar samples) are local-only (the gitignored); the examples
that need them are marked.

## The examples

| Example | What it shows | Needs |
|---|---|---|
| `benchmark.rs` | The naive O(vocab) scan vs the PDA O(1) lookup vs the SIMD bit-pack. The 1300x PDA win + the 93x SIMD win. The 100% accuracy (the 1300/1300 state-token pairs). The {a^n b^n} language check. | none (the synthetic 65-vocab) |
| `bench_realworld.rs` | The real 248K-vocab tokenizer + the real grammars (the JSON-complex, the regex, the xbot). The 4-mode benchmark (the normal, the scalar PDA, the inline SIMD, the device PDA) + the 106/106 accuracy gate. | `bench_realworld/test-tokenizer.json` + `grammar_sample.txt` (the local fixtures) |
| `nom_integration.rs` | The nom parser combinators as the independent oracle. The BER/TLV, the JSON (the nested), the regex. The PDA == the nom parser (the ). | none |
| `deku_integration.rs` | The deku derive-struct parser as the independent oracle. The FixedTlv. The PDA == the deku struct (the differential). | none |
| `pda_router.rs` | The PDA as a network router/switch (the 2-4). The VLAN membership, the route rules, the ACLs. The micromachine data dump. The with/without SIMD comparison. | none |
| `dbg_rtn.rs` | The debug dump of the RTN compilation (the states, the transitions, the trace). | none |
| `viz_dump.rs` | The PDA -> SVG /OT visualization (the viz.rs). The {a^n b^n}, the Dyck, the hand-built DFA. Writes `viz/*.svg` + `viz/*.dot` (the graphviz layout). | none |
| `sql_integration.rs` | The SQL-subset CFG (the left-factored, the DCFL) + the recursive-descent parser (the independent oracle) + the PDA + the differential + the viz dump. The nesting (the Factor -> ( Expr )) exercises the stack. | none |
| `ebpf_integration.rs` | The eBPF well-nested call/return CFG (the DCFL, the the PDA stack tracks the call depth) + the call-depth walker (the independent oracle) + the PDA + the differential + the viz dump. | none |

## Run them

```
   # the all (the no fixtures needed)
   cargo run --release --example benchmark
   cargo run --release --example nom_integration
   cargo run --release --example deku_integration
cargo run --release --example pda_router
    cargo run --release --example dbg_rtn
    cargo run --release --example viz_dump   # the writes viz/*.svg + the/*.dot
    cargo run --release --example sql_integration   # the the SQL-subset PDA + the recursive-descent oracle
    cargo run --release --example ebpf_integration  # the the eBPF call/return PDA + the walker oracle

   # the real-world (the needs the local fixtures)
   cargo run --release --example bench_realworld
```

## The fixtures (the bench_realworld/)

- `test-tokenizer.json` - the 248K-vocab HF tokenizer (the real LLM scale).
- `grammar_sample.txt` - the xbot golden grammar (the tool-call +).
- `METHODOLOGY.md` - the test methodology (the oracles, the corpora, the parameters).
- `RESEARCH.md` - the research (the bench numbers, the accuracy).

These are gitignored (the 20MB tokenizer is not committed). The examples that
need them skip gracefully if the fixtures are absent.