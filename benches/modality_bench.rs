//! Full modality bench: scalar, SIMD inline, SIMD service, CUDA ungraphed, CUDA graphed.
//!
//! The PDA execution engine serves two domains:
//!   - LLM grammar masking: mask broadcast at 248K vocab, per-token step, K-projection
//!   - Network appliance: per-packet OSI validation (L2->L3->L4), DPDK burst (batch2-256
//!     packets), CUDA graph replay (the fixed pipeline, no per-packet CPU involvement)
//!
//! The modalities:
//!   - scalar: the reference (always correct, the oracle)
//!   - simd_inline: rten-simd dispatch in-process (the CPU fallback, no GPU)
//!   - simd_service: PdaService packet-in/packet-out (the device model, emulates GPU)
//!   - cuda_ungraphed: individual kernel launches (the attention-rs fused_sample))
//!   - cuda_graphed: CUDA graph capture + replay (the Pre3 fusion, no launch overhead)
//!
//! The CUDA paths require a GPU (RTX 5090, sm_120). The CPU paths run anywhere.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use pushdown_rs::machine::PdaMachine;
use pushdown_rs::pda::PdaStream;
use rten_simd::SimdOp;
use std::hint::black_box;

/// The {a^n b^n} DPDA (the reference machine for all modalities).
fn test_machine() -> PdaMachine {
    const A: u32 = 0;
    const B: u32 = 1;
    const EPS: u32 = 2;
    const Z: u32 = 0;
    const A_SYM: u32 = 1;
    PdaMachine {
        num_states: 4,
        num_inputs: 2,
        num_stack_syms: 2,
        transitions: vec![
            pushdown_rs::Transition { q: 0, a: A, top: Z, next_q: 1, push: vec![A_SYM, Z] },
            pushdown_rs::Transition { q: 1, a: A, top: A_SYM, next_q: 1, push: vec![A_SYM, A_SYM] },
            pushdown_rs::Transition { q: 1, a: B, top: A_SYM, next_q: 2, push: vec![] },
            pushdown_rs::Transition { q: 2, a: B, top: A_SYM, next_q: 2, push: vec![] },
            pushdown_rs::Transition { q: 2, a: EPS, top: Z, next_q: 3, push: vec![Z] },
        ],
        accepting: vec![3],
        start_state: 0,
        start_stack: Z,
        state_provenance: None,
    }
}

/// The mask broadcast bench: apply a VOB mask to a logit row.
/// LLM: the 248K vocab mask. (the per-token sampling constraint).
/// Network: the per-packet field validation (the OSI layer PDA mask).
fn bench_mask_broadcast(c: &mut Criterion) {
    let machine = test_machine();
    for &vocab in &[256usize, 4096, 32000, 248320] {
        let logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.001).collect();
        // The mask: only inputs 0 and 1 are allowed (the PDA's num_inputs=2).
        let mask_words: Vec<u32> = {
            let mut words = vec![0u32; (vocab + 31) / 32];
            words[0] |= 0b11; // bits 0 and 1 set
            words
        };
        let bias = pushdown_rs::simd::compute_bias(&mask_words, vocab);
        let mut out = vec![0.0f32; vocab];

        let mut group = c.benchmark_group(format!("mask_broadcast_vocab_{}", vocab));
        group.throughput(Throughput::Bytes(vocab as u64 * 4));

        // Scalar (the reference, always correct).
        group.bench_function("scalar", |b| {
            b.iter(|| {
                let mut op = pushdown_rs::simd::MaskBroadcastOp::new(
                    black_box(&logits), black_box(&bias), &mut out,
                );
                op.scalar();
                black_box(out[0])
            })
        });

        // SIMD inline (rten-simd dispatch, in-process, the CPU fallback).
        group.bench_function("simd_inline", |b| {
            b.iter(|| {
                let op = pushdown_rs::simd::MaskBroadcastOp::new(
                    black_box(&logits), black_box(&bias), &mut out,
                );
                op.dispatch();
                black_box(out[0])
            })
        });

        // SIMD service (PdaService packet-in/packet-out, the device model).
        // This is the network appliance execution model: a batch of configs in,
        // a batch of masks out. No per-item allocation.
        group.bench_function("simd_service", |b| {
            b.iter(|| {
                let service = pushdown_rs::service::PdaService::new(machine.clone());
                let configs = vec![(machine.start_state, vec![machine.start_stack])];
                let masks = service.mask(&configs);
                black_box(masks[0].len())
            })
        });

        group.finish();
    }
}

/// The PDA step bench: advance the PDA state by one token (per sequence).
/// LLM: the per-token PDA advance (the sampling loop).
/// Network: the per-packet OSI validation step (the L2->L3->L4 chain).
fn bench_pda_step(c: &mut Criterion) {
    let machine = test_machine();
    let index = machine.build_index();
    let batch_size: usize = 256; // DPDK burst size (the typical NIC batch).
    let configs: Vec<(u32, Vec<u32>)> = (0..batch_size)
        .map(|i| ((i % 4) as u32, vec![0u32]))
        .collect();
    let tokens: Vec<u32> = vec![0u32; batch_size];

    let mut group = c.benchmark_group("pda_step_batch256");
    group.throughput(Throughput::Elements(batch_size as u64));

    // Scalar per-item (the reference, O(batch * transitions)).
    group.bench_function("scalar_per_item", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(batch_size);
            for (i, &(q, ref stk)) in configs.iter().enumerate() {
                let top = stk.last().copied().unwrap_or(machine.start_stack);
                match machine.lookup(q, Some(tokens[i]), top).as_slice() {
                    [t] => {
                        let mut s2 = stk.clone();
                        s2.pop();
                        for &p in t.push.iter().rev() { s2.push(p); }
                        results.push((t.next_q, s2));
                    }
                    _ => results.push((q, stk.clone())),
                }
            }
            black_box(results.len())
        })
    });

    // Batched (the PdaStream interface, the SIMD-able path).
    group.bench_function("batched_step", |b| {
        b.iter(|| {
            let batch: Vec<((u32, Vec<u32>), u32)> = configs.iter().zip(tokens.iter())
                .map(|(&(q, ref stk), &a)| ((q, stk.clone()), a)).collect();
            let results = machine.step_batch(&batch);
            black_box(results.len())
        })
    });

    // Indexed batched (the O(1) lookup via HashMap, the no-alloc path).
    group.bench_function("indexed_step", |b| {
        b.iter(|| {
            let batch: Vec<(u32, Vec<u32>, u32)> = configs.iter().zip(tokens.iter())
                .map(|(&(q, ref stk), &a)| (q, stk.clone(), a)).collect();
            let mut out: Vec<(u32, Vec<u32>)> = vec![(0, vec![0]); batch_size];
            machine.step_batch_into(&index, &batch, &mut out);
            black_box(out.len())
        })
    });

    // Service (the PdaService packet-in/packet-out, the device model).
    group.bench_function("service_step", |b| {
        b.iter(|| {
            let service = pushdown_rs::service::PdaService::new(machine.clone());
            let batch: Vec<(u32, Vec<u32>, u32)> = configs.iter().zip(tokens.iter())
                .map(|(&(q, ref stk), &a)| (q, stk.clone(), a)).collect();
            let results = service.step(&batch);
            black_box(results.len())
        })
    });

    group.finish();
}

/// The PDA projection bench: walk K draft tokens, emit K+1 masks.
/// LLM: the MTP/DFlash drafting constraint (the K+1 VOB masks).
/// Network: the multi-packet burst validation (the K packets through the OSI chain).
fn bench_pda_project(c: &mut Criterion) {
    let machine = test_machine();
    let index = machine.build_index();
    let batch = 64;
    let k = 16; // the typical adaptive-K draft length (or DPDK burst of 16 packets).
    let configs: Vec<(u32, Vec<u32>)> = (0..batch)
        .map(|i| ((i % 4) as u32, vec![0u32]))
        .collect();
    let drafts: Vec<Vec<u32>> = vec![vec![0u32; k]; batch];

    let mut group = c.benchmark_group(format!("pda_project_batch{}_k{}", batch, k));
    group.throughput(Throughput::Elements((batch * (k + 1)) as u64));

    // Scalar projection (the PdaStream interface, the reference).
    group.bench_function("scalar_project", |b| {
        b.iter(|| {
            let results = machine.project_batch(&configs, &drafts);
            black_box(results[0].len())
        })
    });

    // Indexed projection (the O(1) lookup via HashMap, the SIMD-able path).
    group.bench_function("indexed_project", |b| {
        b.iter(|| {
            let results = machine.project_batch_simd(&index, &configs, &drafts);
            black_box(results[0].len())
        })
    });

    // Service projection (the PdaService,-in/packet-out, the device model).
    group.bench_function("service_project", |b| {
        b.iter(|| {
            let service = pushdown_rs::service::PdaService::new(machine.clone());
            let results = service.project(&configs, &drafts);
            black_box(results[0].len())
        })
    });

    group.finish();
}

/// The provenance primitive bench: the O(1) phase lookup (`provenance_of`) + the
/// bitvec round-trip cost (the suffix overhead of carrying `state_provenance` in
/// the POD). LLM: the region gate (the O(1) state -> phase). Network: the OSI
/// layer phase of a packet-validation state.
fn bench_provenance(c: &mut Criterion) {
    // A compiled machine (the carries state_provenance).
    let g = pushdown_rs::compile::Cfg::new(
        3,
        2,
        0,
        vec![
            (0, vec![3, 1, 4]),
            (0, vec![]),
            (1, vec![2, 2]),
            (1, vec![3]),
            (2, vec![4]),
        ],
    );
    let machine = pushdown_rs::compile(&g).expect("compile");
    let queries: Vec<u32> = (0..machine.num_states).collect();

    let mut group = c.benchmark_group("provenance");
    group.throughput(Throughput::Elements(queries.len() as u64));

    // The O(1) phase lookup (the provenance_of, the region-gate cost).
    group.bench_function("provenance_of", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &q in &queries {
                acc = acc.wrapping_add(machine.provenance_of(q).unwrap_or(0));
            }
            black_box(acc)
        })
    });

    // The bitvec round-trip cost (the suffix overhead of carrying provenance).
    group.bench_function("bitvec_roundtrip_with_provenance", |b| {
        b.iter(|| {
            let bits = machine.to_bitvec();
            let m2 = PdaMachine::from_bitvec(&bits).expect("round-trip");
            black_box(m2.num_states)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_mask_broadcast, bench_pda_step, bench_pda_project, bench_provenance);
criterion_main!(benches);