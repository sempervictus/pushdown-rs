//! Criterion benches for the PDA hot-path ops (the CSR mask + the pass-through).
//!
//! Measures, on a nested tool-call-style CFG (the bounded-stack DCFL shape):
//!   - the CSR-based settled mask (the O(1-3) mask_at_cfg_settled) vs the
//!     independent linear reference (the O(num_inputs x total) scan) - the
//!     speedup from the CSR fix.
//!   - the CSR-based epsilon-closure mask (the mask_at_cfg).
//!   - the advance_eps (the O(1-3) lockstep step).
//!   - the passthrough_run (the linear-run length).
//!
//! The key property (the six-property "bounded control"): the per-step PDA cost
//! is O(1) in the INPUT length (the config space is bounded by kappa(G)), so
//! these ops are flat as the input grows (the no unbounded item-set growth).

use criterion::{criterion_group, criterion_main, Criterion, Throughput, black_box};
use pushdown_rs::compile::Cfg;
use pushdown_rs::machine::PdaMachine;
use pushdown_rs::pda::PdaStream;

// A nested tool-call-style CFG (the DCFL, the bounded stack):
//   S    -> LBRACE args RBRACE
//   args -> str COLON val COMMA args | str COLON val
//   str  -> QUOTE STR QUOTE
//   val  -> QUOTE NUM QUOTE | S
// The Cfg symbol convention: the nonterminals are 0..num_nonterminals, the
// terminals are num_nonterminals..(num_nonterminals + num_terminals). With
// 4 nonterminals (S=0, args=1, str=2, val=3) + 7 terminals, the terminal IDs
// are 4..11: LBRACE=4, RBRACE=5, COLON=6, COMMA=7, QUOTE=8, STR=9, NUM=10.
fn tool_call_cfg() -> Cfg {
    const S: u32 = 0;
    const ARGS: u32 = 1;
    const STR: u32 = 2;
    const VAL: u32 = 3;
    const LBRACE: u32 = 4;
    const RBRACE: u32 = 5;
    const COLON: u32 = 6;
    const COMMA: u32 = 7;
    const QUOTE: u32 = 8;
    const STR_T: u32 = 9;
    const NUM: u32 = 10;
    Cfg::new(
        4, // S, ARGS, STR, VAL
        7, // LBRACE..NUM (the 4..10)
        S,
        vec![
            (S, vec![LBRACE, ARGS, RBRACE]), // S -> { args }
            (ARGS, vec![STR, COLON, VAL, COMMA, ARGS]), // args -> str : val , args
            (ARGS, vec![STR, COLON, VAL]), // args -> str : val
            (STR, vec![QUOTE, STR_T, QUOTE]), // str -> " STR "
            (VAL, vec![QUOTE, NUM, QUOTE]), // val -> " NUM "
            (VAL, vec![S]), // val -> S (the nested object)
        ],
    )
}

// The independent linear reference for the settled mask (the O(num_inputs x
// total_transitions) scan, the NOT the CSR). The oracle for the speedup.
fn linear_settled(m: &PdaMachine, q: u32, top: u32) -> Vec<u32> {
    let mut allowed = Vec::new();
    for a in 0..m.num_inputs {
        if !m.lookup(q, Some(a), top).is_empty() {
            allowed.push(a);
        }
    }
    allowed
}

fn bench_csr_vs_linear_mask(c: &mut Criterion) {
    let g = tool_call_cfg();
    let m = pushdown_rs::compile(&g).expect("compile");
    assert!(!m.ctrl_offsets.is_empty(), "the CSR must be present");
    // A representative config: the start (the S -> { args } entry).
    let q = m.start_state;
    let top = m.start_stack;
    let stack = vec![m.start_stack];

    let mut group = c.benchmark_group("pda_mask");
    group.throughput(Throughput::Elements(1));
    group.bench_function("csr_settled", |b| {
        b.iter(|| black_box(m.mask_at_cfg_settled(q, top)))
    });
    group.bench_function("linear_settled_ref", |b| {
        b.iter(|| black_box(linear_settled(&m, q, top)))
    });
    group.bench_function("csr_epsilon_closure", |b| {
        b.iter(|| black_box(m.mask_at_cfg(q, &stack)))
    });
    group.bench_function("advance_eps", |b| {
        b.iter(|| black_box(m.advance_eps(q, &stack, 0)))
    });
    group.bench_function("passthrough_run", |b| {
        b.iter(|| black_box(m.passthrough_run(q)))
    });
    group.finish();
}

fn bench_sequence_length_flat(c: &mut Criterion) {
    // The key property: the per-step PDA cost is O(1) in the input length (the
    // bounded config space). We walk a long legal derivation and measure the
    // batched projection cost (the flat, the no growth).
    let g = tool_call_cfg();
    let m = pushdown_rs::compile(&g).expect("compile");
    let k = 1024;
    // Build a legal input by walking advance_eps on the first available input.
    // The available inputs at a config are the EPSILON-CLOSURE mask (mask_at_cfg):
    // from q_start the PDA must do epsilon moves (the start + the choice) before
    // it can consume, so the settled mask (mask_at_cfg_settled) is empty there and
    // would break the walk at step 0. mask_at_cfg is exactly the set for which
    // advance_eps succeeds (the proof_mask_batch_consistent_with_advance_eps).
    let mut cur = (m.start_state, vec![m.start_stack]);
    let mut draft: Vec<u32> = Vec::with_capacity(k);
    for _ in 0..k {
        let mask = m.mask_at_cfg(cur.0, &cur.1);
        let Some(a) = mask.first().copied() else {
            break;
        };
        draft.push(a);
        match m.advance_eps(cur.0, &cur.1, a) {
            Some(nc) => cur = nc,
            None => break,
        }
    }
    let walked = draft.len();
    let configs = vec![(m.start_state, vec![m.start_stack])];

    let mut group = c.benchmark_group(format!("pda_seqflat_walked{walked}"));
    group.throughput(Throughput::Elements(walked as u64));
    group.bench_function("project_batch", |b| {
        b.iter(|| {
            let _ = m.project_batch(&configs, &[draft.clone()]);
            black_box(())
        })
    });
    group.finish();
}

    fn bench_dispatched_gather(c: &mut Criterion) {
    // The dispatched csr_gather (the B lanes in chunks of the SIMD width, the
    // fearless_simd dispatch!). The B is big (the 256, the 512-bit pipeline
    // loaded). The independent linear reference is the O(num x total) scan.
    use pushdown_rs::simd_pipeline::csr_gather;
    let g = tool_call_cfg();
    let m = pushdown_rs::compile(&g).expect("compile");
    let b = 256;
    let ctrls: Vec<u32> = (0..b).map(|i| (i % m.num_states) as u32).collect();
    let tops: Vec<u32> = (0..b).map(|i| (i % m.num_stack_syms) as u32).collect();

    fn linear_gather(m: &PdaMachine, ctrls: &[u32], tops: &[u32]) -> Vec<Vec<u32>> {
        ctrls.iter().zip(tops.iter()).map(|(&q, &top)| {
            let mut allowed = Vec::new();
            for a in 0..m.num_inputs {
                if !m.lookup(q, Some(a), top).is_empty() {
                    allowed.push(a);
                }
            }
            allowed
        }).collect()
    }

    let mut group = c.benchmark_group("pda_gather");
    group.throughput(Throughput::Elements(b as u64));
    group.bench_function("dispatched_csr_gather", |it| {
        it.iter(|| black_box(csr_gather(&m, &ctrls, &tops)))
    });
    group.bench_function("linear_gather_ref", |it| {
        it.iter(|| black_box(linear_gather(&m, &ctrls, &tops)))
    });
    group.finish();
}

criterion_group!(benches, bench_csr_vs_linear_mask, bench_sequence_length_flat, bench_dispatched_gather);
criterion_main!(benches);