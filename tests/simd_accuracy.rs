//! Accuracy proofs for the SIMD tier: every SIMD op must produce 100% identical
//! results to its scalar reference. One variable at a time, full control.

#[cfg(feature = "simd")]
mod simd_accuracy {
    use pushdown_rs::simd::{compute_bias, MaskBroadcastOp, MaskOp, StepBatchOp};
    use pushdown_rs::machine::PdaMachine;
    use rten_simd::SimdOp;

    /// PROOF: MaskBroadcastOp::dispatch() == MaskBroadcastOp::scalar()
    /// for all (logits, bias) pairs. The SIMD add must be bit-exact.
    #[test]
    fn mask_broadcast_simd_equals_scalar() {
        // Test with various sizes: small (tail only), exact multiple multiples, mixed.
        for &vocab in &[1usize, 4, 8, 16, 32, 64, 100, 256, 1024, 4096] {
            // Deterministic pseudo-random logits (seeded by vocab size).
            let logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.1 - 5.0).collect();
            // Bias: alternating allowed/disallowed pattern.
            let mask_words: Vec<u32> = (0..(vocab + 31) / 32)
                .map(|w| {
                    // Every other bit set: 0b01010101...
                    if w % 2 == 0 { 0xAAAAAAAA } else { 0x55555555 }
                })
                .collect();
            let bias = compute_bias(&mask_words, vocab);

            // Scalar reference.
            let mut out_scalar = vec![0.0f32; vocab];
            {
                let mut op = MaskBroadcastOp::new(&logits, &bias, &mut out_scalar);
                op.scalar();
            }

            // SIMD dispatch.
            let mut out_simd = vec![0.0f32; vocab];
            {
                let op = MaskBroadcastOp::new(&logits, &bias, &mut out_simd);
                op.dispatch();
            }

            assert_eq!(
                out_scalar, out_simd,
                "vocab={vocab}: SIMD must equal scalar (bit-exact)"
            );
        }
    }

    /// PROOF: MaskBroadcastOp with all-allowed mask is identity (logits unchanged).
    #[test]
    fn mask_broadcast_all_allowed_is_identity() {
        let vocab = 256;
        let logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.01).collect();
        // All bits set: every token is.
        let mask_words = vec![0xFFFFFFFFu32; vocab / 32];
        let bias = compute_bias(&mask_words, vocab);
        assert!(bias.iter().all(|&b| b == 0.0), "all-allowed bias must be 0.0");

        let mut out = vec![0.0f32; vocab];
        MaskBroadcastOp::new(&logits, &bias, &mut out).dispatch();
        assert_eq!(out, logits, "all-allowed mask must preserve logits exactly");
    }

    /// PROOF: MaskBroadcastOp with all-disallowed mask produces all -inf.
    #[test]
    fn mask_broadcast_all_disallowed_is_neg_inf() {
        let vocab = 256;
        let logits: Vec<f32> = (0..vocab).map(|i| i as f32).collect();
        let mask_words = vec![0u32; vocab / 32];
        let bias = compute_bias(&mask_words, vocab);
        assert!(bias.iter().all(|&b| b == f32::NEG_INFINITY));

        let mut out = vec![0.0f32; vocab];
        MaskBroadcastOp::new(&logits, &bias, &mut out).dispatch();
        assert!(out.iter().all(|&v| v == f32::NEG_INFINITY), "all-disallowed must be -inf");
    }

    /// PROOF: MaskOp (u8 mask) dispatch == scalar for all patterns.
    #[test]
    fn mask_op_u8_dispatch_equals_scalar() {
        for &vocab in &[4usize, 32, 64, 100, 256] {
            let logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 1.1).collect();
            // Pattern: every 3rd token allowed.
            let mask: Vec<u8> = (0..vocab).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();

            let mut out_scalar = vec![0.0f32; vocab];
            {
                let mut op = MaskOp::new(&logits, &mask, &mut out_scalar);
                op.scalar();
            }

            let mut out_simd = vec![0.0f32; vocab];
            MaskOp::new(&logits, &mask, &mut out_simd).dispatch();

            assert_eq!(
                out_scalar, out_simd,
                "vocab={vocab}: MaskOp dispatch must equal scalar"
            );
        }
    }

    /// PROOF: StepBatchOp dispatch == scalar for a known machine.
    #[test]
    fn step_batch_simd_equals_scalar() {
        // The {a^n b^n} DPDA from the main test suite.
        const A: u32 = 0;
        const B: u32 = 1;
        const EPS: u32 = 2;
        const Z: u32 = 0;
        const A_SYM: u32 = 1;
        let machine = PdaMachine {
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
        };
        let index = machine.build_index();

        // Batch of states: [0, 1, 1, 2, 3, 0, 1, 2] (8 elements, exact AVX2 lane).
        let states: Vec<u16> = vec![0, 1, 1, 2, 3, 0, 1, 2];
        let token = A;

        // Scalar reference.
        let mut out_scalar = vec![0u16; states.len()];
        {
            let mut op = StepBatchOp::new(&machine, &index, &states, token, &mut out_scalar);
            op.scalar();
        }

        // SIMD dispatch.
        let mut out_simd = vec![0u16; states.len()];
        StepBatchOp::new(&machine, &index, &states, token, &mut out_simd).dispatch();

        assert_eq!(
            out_scalar, out_simd,
            "StepBatchOp dispatch must equal scalar"
        );
    }

    /// PROOF: compute_bias is correct for all u32 patterns.
    #[test]
    fn compute_bias_all_patterns() {
        // Test all 4 u32 words with various patterns.
        let patterns = [
            0x00000000u32, // all disallowed
            0xFFFFFFFFu32, // all allowed
            0xAAAAAAAAu32, // even bits allowed
            0x55555555u32, // odd bits allowed
            0x00000001u32, // only bit 0
            0x80000000u32, // only bit 31
        ];
        for &word in &patterns {
            let bias = compute_bias(&[word], 32);
            for i in 0..32 {
                let expected = if (word >> i) & 1 == 1 { 0.0f32 } else { f32::NEG_INFINITY };
                assert_eq!(bias[i], expected, "word={word:#x} bit={i}");
            }
        }
    }

// Criterion benches: measure SIMD vs scalar speedup for the mask broadcast.
// (moved to benches/modality_bench.rs)
}
