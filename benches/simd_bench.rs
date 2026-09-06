//! Criterion benches for the SIMD mask broadcast.
//! Measures scalar vs SIMD dispatch at realistic vocab sizes.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use pushdown_rs::simd::{compute_bias, MaskBroadcastOp};
use rten_simd::SimdOp;
use std::hint::black_box;

fn bench_mask_broadcast(c: &mut Criterion) {
    // Realistic vocab sizes: 256 (small), 4096 (medium), 32000 (LLM), 248320 (248K).
    for &vocab in &[256usize, 4096, 32000, 248320] {
        let logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.001).collect();
        // 50% allowed pattern.
        let mask_words: Vec<u32> = vec![0x55555555u32; (vocab + 31) / 32];
        let bias = compute_bias(&mask_words, vocab);
        let mut out = vec![0.0f32; vocab];

        let mut group = c.benchmark_group(format!("mask_broadcast_vocab_{vocab}"));
        group.throughput(Throughput::Bytes(vocab as u64 * 4));

        group.bench_function("scalar", |b| {
            b.iter(|| {
                let mut op = MaskBroadcastOp::new(black_box(&logits), black_box(&bias), &mut out);
                op.scalar();
                black_box(out[0])
            })
        });

        group.bench_function("simd_dispatch", |b| {
            b.iter(|| {
                let op = MaskBroadcastOp::new(black_box(&logits), black_box(&bias), &mut out);
                op.dispatch();
                black_box(out[0])
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_mask_broadcast);
criterion_main!(benches);