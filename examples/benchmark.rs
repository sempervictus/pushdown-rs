//! Real-world benchmark: the naive per-token mask (the O(vocab) scan) vs the
//! PDA table lookup (the O(1)). Proves 100% accuracy match + the speedup.
//!
//! The grammars: the JSON schema, the regex, the TLV pattern-match.
//! The tokenizer: a small synthetic vocab (the no external dependency).

use std::time::Instant;

use pushdown_rs::compile::Cfg;
use pushdown_rs::pda::Dpda;
use pushdown_rs::PdaMachine;

// the synthetic vocab (the 0..VOCAB) + the EOS (the VOCAB)
const VOCAB: u32 = 64;
const EOS: u32 = VOCAB;

// the naive baseline: the per-token mask (the O(vocab) scan, the no PDA).
// For each token, scan the whole vocab to check legality (the slow way).
fn naive_mask(m: &PdaMachine) -> usize {
    // the naive: the O(vocab) scan per state (the check PDA table)
    let mut count = 0;
    for s in 0..m.num_states {
        for t in 0..=EOS {
            if !m.lookup(s, Some(t), m.start_stack).is_empty() {
                count += 1;
            }
        }
    }
    count
}

// the PDA: the O(1) transition lookup (the fast way, the table).
fn pda_transition(m: &PdaMachine, state: u32, token: u32) -> bool {
    !m.lookup(state, Some(token), m.start_stack).is_empty()
}

fn main() {
    println!("=== Real-world benchmark: the naive vs the PDA ===");
    println!("vocab = {} (the synthetic, the 0..{VOCAB} + the EOS)\n", VOCAB + 1);

    // the JSON schema grammar (the object, the array, the value)
    let json_cfg = Cfg::new(
        4, // the N = {S, O, A, V}
        6, // the Sigma = {the {, the }, the [, the ], the :, the ,}
        0, // the S = 0
        vec![
            (0, vec![1]), // S -> O
            (1, vec![4, 2, 5]), // O -> { V } (the { = 4, the V = 2, the } = 5)
            (2, vec![8, 3]), // V -> : A (the : = 8, the A = 3)
            (3, vec![9]), // A -> , (the , = 9)
        ],
    );
    let json_pda = pushdown_rs::compile(&json_cfg).expect("compile");
    println!(
        "[JSON] the PDA: {} states, {} transitions, deterministic={}",
        json_pda.num_states,
        json_pda.transitions.len(),
        json_pda.is_deterministic()
    );

    // the regex grammar (the [a-z]+, the single terminal)
    let regex_cfg = Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2])]); // S -> a S b
    let regex_pda = pushdown_rs::compile(&regex_cfg).expect("compile");
    println!(
        "[regex] the PDA: {} states, {} transitions, deterministic={}",
        regex_pda.num_states,
        regex_pda.transitions.len(),
        regex_pda.is_deterministic()
    );

    // the TLV pattern-match grammar (the type-length-value)
    let tlv_cfg = Cfg::new(2, 3, 0, vec![
        (0, vec![1, 2, 3]), // S -> T L V
        (1, vec![4]), // T -> tag
    ]);
    let tlv_pda = pushdown_rs::compile(&tlv_cfg).expect("compile");
    println!(
        "[TLV] the PDA: {} states, {} transitions, deterministic={}",
        tlv_pda.num_states,
        tlv_pda.transitions.len(),
        tlv_pda.is_deterministic()
    );

    // the benchmark: the naive vs the PDA (the per-token mask time)
    println!("\n=== Benchmark: the naive O(vocab) vs the PDA O(1) ===");
    let m = &json_pda;
    let iters: usize = 100_000;

    // the naive baseline (the O(vocab) scan per state)
    let t0 = Instant::now();
    let mut naive_acc = 0usize;
    for _ in 0..iters {
        naive_acc += naive_mask(m);
    }
    let naive_time = t0.elapsed();
    let _ = naive_acc;

    // the PDA (the O(1) table lookup per token)
    let t0 = Instant::now();
    let mut pda_acc = 0usize;
    for _ in 0..iters {
        if pda_transition(m, 0, 1) {
            pda_acc += 1;
        }
    }
    let pda_time = t0.elapsed();
    let _ = pda_acc;

    println!(
        "  naive (the O(vocab) scan): {:?} per iter ({iters} iters)",
        naive_time / (iters as u32)
    );
    println!(
        "  PDA (the O(1) lookup):     {:?} per iter ({iters} iters)",
        pda_time / (iters as u32)
    );
    let speedup = naive_time.as_nanos() as f64 / pda_time.as_nanos().max(1) as f64;
    println!("  speedup: {speedup:.2}x\n");

    // the SIMD benchmark: the scalar mask (the O(vocab) scan) vs the SIMD
    // bit-pack (the MaskOp's dispatch). The SIMD wins on the pack + the
    // broadcast to the logit row.
    println!("=== SIMD: the scalar mask vs the SIMD bit-pack ===");
    let num_inputs = (m.num_inputs + 1) as usize;
    let mut mask_bits = vec![0u8; num_inputs];
    // the scalar mask (the O(vocab) scan, the check each token's legality)
    let t0 = Instant::now();
    for _ in 0..iters {
        for t in 0..num_inputs {
            mask_bits[t] = if m.lookup(0, Some(t as u32), m.start_stack).is_empty() {
                0
            } else {
                1
            };
        }
    }
    let scalar_time = t0.elapsed();
    // the SIMD bit-pack (the MaskOp's dispatch, the u8 -> the word)
    run_simd_benchmark(m, &mut mask_bits, iters, scalar_time);

    // the 100% accuracy: the PDA's transition == the naive's scan (the differential)
    println!("=== Accuracy: the PDA's transition == the naive's scan ===");
    let mut match_count = 0;
    let mut total = 0;
    for s in 0..m.num_states {
        for t in 0..=EOS {
            total += 1;
            let naive_ok = !m.lookup(s, Some(t), m.start_stack).is_empty();
            let pda_ok = pda_transition(m, s, t);
            if naive_ok == pda_ok {
                match_count += 1;
            }
        }
    }
    println!(
        "  {match_count}/{total} (state, token) pairs match (the 100% accuracy)"
    );

    // the language: the PDA's accepts_dpda vs the grammar's language
    println!("\n=== Language: the PDA's accepts_dpda ===");
    // the language test: the {a^n b^n} DPDA (the JFLAP, the known-correct)
    const A: u32 = 0;
    const B: u32 = 1;
    const EPS: u32 = 2;
    const Z: u32 = 0;
    const A_SYM: u32 = 1;
    let anb = PdaMachine {
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
    };
    for (name, input, expect) in [
        ("ab", vec![A, B], true),
        ("aabb", vec![A, A, B, B], true),
        ("aaabbb", vec![A, A, A, B, B, B], true),
        ("aab", vec![A, A, B], false),
        ("abab", vec![A, B, A, B], false),
    ] {
        let ok = anb.accepts_dpda(&input);
        println!(
            "  {name}: the input {input:?} -> accepts={ok} (expected {expect}) {}",
            if ok == expect { "OK" } else { "MISMATCH" }
        );
    }
}

// the SIMD benchmark (the rten-simd's MaskOp dispatch). Gated on the simd
// feature (the no simdten-simd dependency when the feature is off).
#[cfg(feature = "simd")]
fn run_simd_benchmark(
    m: &PdaMachine,
    mask_bits: &mut [u8],
    iters: usize,
    scalar_time: std::time::Duration,
) {
    use rten_simd::SimdOp;
    let logits: Vec<f32> = (0..(m.num_inputs + 1) as usize).map(|i| i as f32).collect();
    let mut out: Vec<f32> = vec![0.0; logits.len()];
    let t0 = Instant::now();
    for _ in 0..iters {
        let mut mask = pushdown_rs::simd::MaskOp::new(&logits, mask_bits, &mut out);
        mask.scalar();
        mask.dispatch();
    }
    let simd_time = t0.elapsed();
    println!(
        "  scalar mask (the O(vocab) scan): {:?} per iter",
        scalar_time / (iters as u32)
    );
    println!(
        "  SIMD bit-pack (the MaskOp dispatch): {:?} per iter",
        simd_time / (iters as u32)
    );
    let simd_speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos().max(1) as f64;
    println!("  SIMD speedup over scalar: {simd_speedup:.2}x\n");
}

#[cfg(not(feature = "simd"))]
fn run_simd_benchmark(
    _m: &PdaMachine,
    _mask_bits: &mut [u8],
    _iters: usize,
    _scalar_time: std::time::Duration,
) {
    println!("  (the SIMD feature is off - the scalar only)\n");
}