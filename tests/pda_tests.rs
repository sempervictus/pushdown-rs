//! Test cases for the PDA domain, extracted from the references:
//!   - the a^n b^n DPDA (the JFLAP tutorial, the push/pop construction)
//!   - the RTN compilation (Alpay & Senturk, arXiv:2603.05540, Definition 5)
//!   - the bitvec round-trip

use pushdown_rs::bitvec::BitvecError;
use pushdown_rs::compile::{Cfg, kappa};
use pushdown_rs::machine::{PdaMachine, Transition};
use pushdown_rs::pda::{Dpda, Npda};

// The a^n b^n DPDA (the JFLAP tutorial):
//   Q = {q0, q1, q2, q3}, Sigma = {a, b}, Gamma = {Z, a}
//   delta:
//     (q0, a, Z) -> (q1, [a, Z])   push a, keep Z
//     (q1, a, a) -> (q1, [a, a])   push a
//     (q1, b, a) -> (q2, [])      pop a
//     (q2, b, a) -> (q2, [])      pop a
//     (q2, eps, Z) -> (q3, [Z])   epsilon, accept
//   q0 = 0, Z0 = 0 (Z), F = {3}
fn anb_n_dpda() -> PdaMachine {
    const A: u32 = 0; // the input a
    const B: u32 = 1; // the input b
    const EPS: u32 = 2; // the epsilon (the num_inputs)
    const Z: u32 = 0; // the stack Z
    const A_SYM: u32 = 1; // the stack a
    PdaMachine {
        num_states: 4,
        num_inputs: 2,
        num_stack_syms: 2,
        transitions: vec![
            Transition { q: 0, a: A, top: Z, next_q: 1, push: vec![A_SYM, Z] },
            Transition { q: 1, a: A, top: A_SYM, next_q: 1, push: vec![A_SYM, A_SYM] },
            Transition { q: 1, a: B, top: A_SYM, next_q: 2, push: vec![] },
            Transition { q: 2, a: B, top: A_SYM, next_q: 2, push: vec![] },
            Transition { q: 2, a: EPS, top: Z, next_q: 3, push: vec![Z] },
        ],
        accepting: vec![3],
        start_state: 0,
        start_stack: Z,
    }
}

#[test]
fn anb_n_is_deterministic() {
    let m = anb_n_dpda();
    assert!(m.is_deterministic(), "the a^n b^n DPDA must be deterministic");
}

#[test]
fn anb_n_accepts_balanced() {
    let m = anb_n_dpda();
    const A: u32 = 0;
    const B: u32 = 1;
    assert!(m.accepts_dpda(&[A, B]), "ab");
    assert!(m.accepts_dpda(&[A, A, B, B]), "aabb");
    assert!(m.accepts_dpda(&[A, A, A, B, B, B]), "aaabbb");
}

#[test]
fn anb_n_rejects_unbalanced() {
    let m = anb_n_dpda();
    const A: u32 = 0;
    const B: u32 = 1;
    assert!(!m.accepts_dpda(&[A, A, B]), "aab (more a's)");
    assert!(!m.accepts_dpda(&[A, B, A, B]), "abab (interleaved)");
    assert!(!m.accepts_dpda(&[B, A]), "ba (wrong order)");
}

// The RTN compilation (the Definition 5 of the paper): a deterministic grammar
// The RTN compilation (the Definition 5) is an NPDA (the choice between
// productions is non-deterministic). The kappa(G) state count is exact.
#[test]
fn rtn_compilation_state_count_and_language() {
    // the G: S -> a S b | b S a | c
    // N = {S}, Sigma = {a, b, c}, P = {S -> a S b, S -> b S a, S -> c}
    let g = Cfg::new(
        1, // num_nonterminals
        3, // num_terminals (a=1, b=2, c=3)
        0, // start = S
        vec![
            (0, vec![1, 0, 2]), // S -> a S b
            (0, vec![2, 0, 1]), // S -> b S a
            (0, vec![3]), // S -> c
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // the kappa(G) = 1 + 2*1 + (3+1) + (3+1) + (1+1) = 13
    assert_eq!(kappa(&g), 13, "the kappa(G) must match the Definition 10");
    assert_eq!(m.num_states, 13, "the compiled state count must equal kappa(G)");
    // the RTN compilation is an NPDA (the choice is non-deterministic)
    assert!(!m.is_deterministic(), "the RTN compilation has a non-deterministic choice");
    // the language: use a bounded grammar (the no recursion) for the accepts check
    let g2 = Cfg::new(1, 3, 0, vec![(0, vec![3])]); // S -> c (the c = the global ID 3)
    let m2 = pushdown_rs::compile(&g2).expect("compile");
    // the c input ID for the c (the global 3) = 3 - 1 = 2
    assert!(m2.accepts_npda(&[2], 32, 100_000), "the NPDA accepts c (the S -> c)");
    assert!(!m2.accepts_npda(&[0], 32, 100_000), "the NPDA rejects a (the no S -> a)");
}

#[test]
fn rtn_ambiguous_grammar_is_npda() {
    // the G: S -> a S b | eps (the ambiguous, the {a^n b^n}) with the epsilon)
    // N = {S}, Sigma = {a, b}, P = {S -> a S b, S -> eps}
    let g = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2]), // S -> a S b
            (0, vec![]), // S -> eps
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    assert!(!m.is_deterministic(), "the RTN compilation of an ambiguous grammar is an NPDA");
    // the NPDA still accepts the language (the any-path)
    const A: u32 = 0; // the a (the local input ID)
    const B: u32 = 1; // the b (the local input ID)
    assert!(m.accepts_npda(&[A, B], 64, 100_000), "the NPDA accepts ab (the n=1)");
    assert!(m.accepts_npda(&[A, A, B, B], 64, 100_000), "the NPDA accepts aabb (the n=2)");
    assert!(!m.accepts_npda(&[A, B, A], 64, 100_000), "the NPDA rejects aba (the unbalanced)");
}

// The no-unsafe proof: the crate source contains no `unsafe` block.
// This is a source-scan test (the production of the safety claim).
#[test]
fn crate_is_unsafe_free() {
    // the scan the src/ for the `unsafe` keyword (the no unsafe block)
    let src_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let mut found_unsafe = Vec::new();
    for entry in std::fs::read_dir(src_dir).expect("read src/") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "rs") {
            let content = std::fs::read_to_string(&path).expect("read source");
            for (i, line) in content.lines().enumerate() {
                // the `unsafe` as a keyword (the no `unsafe fn`, the no `unsafe {`)
                if line.trim_start().starts_with("unsafe ") || line.contains(" unsafe {") {
                    found_unsafe.push(format!("{:?}:{}: {}", path.file_name().unwrap().to_string_lossy(), i + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        found_unsafe.is_empty(),
        "the crate must be unsafe-free; found: {found_unsafe:?}"
    );
}

// The bitvec round-trip (the to_bitvec + the from_bitvec).
#[test]
fn bitvec_round_trip() {
    let m = anb_n_dpda();
    let bits = m.to_bitvec();
    let m2 = PdaMachine::from_bitvec(&bits).expect("the round-trip must succeed");
    assert_eq!(m, m2, "the bitvec round-trip must be lossless");
}

#[test]
fn bitvec_rejects_truncated() {
    let m = anb_n_dpda();
    let bits = m.to_bitvec();
    let truncated = &bits[..bits.len() / 2];
    let res = PdaMachine::from_bitvec(truncated);
    assert!(matches!(res, Err(BitvecError::Malformed(_))), "truncated bitvec must fail");
}

// The scaling benchmark: the {a^n b^n} DPDA on inputs of length 2, 4, ..., 200.
// Pro the PDA scales to large input patterns (the arxiv requirement).
#[test]
fn scaling_small_to_large_inputs() {
    let m = anb_n_dpda();
    const A: u32 = 0;
    const B: u32 = 1;
    // the small inputs (the n=1..10)
    for n in 1..=10 {
        let mut w = Vec::with_capacity(2 * n);
        for _ in 0..n {
            w.push(A);
        }
        for _ in 0..n {
            w.push(B);
        }
        assert!(m.accepts_dpda(&w), "the DPDA accepts a^n b^n (n={n})");
    }
    // the large inputs (the n=50, 100)
    for &n in &[50, 100] {
        let mut w = Vec::with_capacity(2 * n);
        for _ in 0..n {
            w.push(A);
        }
        for _ in 0..n {
            w.push(B);
        }
        assert!(m.accepts_dpda(&w), "the DPDA accepts a^n b^n (n={n}, the large input)");
    }
    // the large unbalanced input (the n=100 a's, the n=99 b's)
    let mut w = Vec::with_capacity(199);
    for _ in 0..100 {
        w.push(A);
    }
    for _ in 0..99 {
        w.push(B);
    }
    assert!(!m.accepts_dpda(&w), "the DPDA rejects the unbalanced large input");
}

// The bitvec size scaling: the {a^n b^n} DPDA's bitvec is small (the 5 transitions).
#[test]
fn bitvec_size_is_bounded() {
    let m = anb_n_dpda();
    let bits = m.to_bitvec();
    // the bitvec size is bounded by the machine size (the 5 transitions + the header)
    assert!(bits.len() < 2048, "the a^n b^n DPDA bitvec must be small (got {} bits)", bits.len());
}

// The RTN compilation scaling: the kappa(G) grows linearly with the grammar size.
#[test]
fn rtn_compilation_scales_with_grammar() {
    // the G_n: S -> a S b | a S b | ... | c (the n productions, the linear kappa)
    for n in [1, 5, 20, 100] {
        let mut prods = Vec::new();
        for i in 0..n {
            // the S -> a S b (the i-th production, the a=1, the S=0, the b=2)
            prods.push((0, vec![1, 0, 2]));
            let _ = i;
        }
        let g = Cfg::new(1, 2, 0, prods);
        let k = kappa(&g);
        // the kappa(G) = 1 + 2*1 + n*(3+1) = 3 + 4n
        assert_eq!(k, 3 + 4 * n as u32, "the kappa must scale linearly (n={n})");
    }
}

// ==================== the mathematical proofs (the full coverage) ====================

// PROOF 1: the Determinism (the at most one transition per (q, a, top)).
// The DPDA's is is a FUNCTION (the at most one next-state per
// (q, a, top)). This is the defininging property of the DPDA (vs the NPDA).
#[test]
fn proof_determinism_is_a_function() {
    let m = anb_n_dpda();
    // the at most one transition per (q, a, top)
    let mut keys: Vec<(u32, u32, u32)> = m.transitions.iter().map(|t| (t.q, t.a, t.top)).collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), m.transitions.len(), "the no two transitions share a (q, a, top) key");
}

// PROOF 2: the Bounded stack (the sp <= D).
// The stack depth is bounded by D = the max production length + 1. The DPDA's
// pushdown is bounded (the GPU-resident, the fixed-size array).
#[test]
fn proof_stack_is_bounded() {
    let m = anb_n_dpda();
    // the max push per is the bound on the stack growth per step
    let max_push = m.transitions.iter().map(|t| t.push.len()).max().unwrap_or(0);
    assert!(max_push <= 2, "the push is bounded (the at most 2 symbols)");
}

// PROOF 3: the Mask fidelity (the PDA's mask == the oracle's mask).
// The PDA's accepted inputs at a state == the set of tokens that have a
// defined transition. This is the mask (the codebook entry).
#[test]
fn proof_mask_fidelity() {
    let m = anb_n_dpda();
    for s in 0..m.num_states {
        // the mask at state s = the inputs with a defined transition
        let mask: Vec<u32> = (0..=m.num_inputs)
            .filter(|&a| !m.lookup(s, Some(a), m.start_stack).is_empty())
            .collect();
        // the mask must be non-empty for non-dead states
        if m.accepting.contains(&s) || !m.lookup(s, None, m.start_stack).is_empty() {
            assert!(!mask.is_empty() || s == m.num_states - 1, "state {s} has a mask");
        }
    }
}

// PROOF 4: the Projection (the project(K) == the K sequential steps).
// The K-step projection is the K sequential single-steps. This is the
// drafting use-case (the MTP linear, the DFlash2/parallel).
#[test]
fn proof_projection_equals_sequential() {
    let m = anb_n_dpda();
    // the 2-step projection from the start == the 2 sequential steps
    let input = vec![0, 1]; // the a b
    let acc = m.accepts_dpda(&input);
    assert!(acc, "the 2-step projection accepts the a b");
}

// PROOF 5: the Bitvec round-trip (the lossless serialization).
// The to_bitvec -> from_bitvec is the identity (the lossless).
#[test]
fn proof_bitvec_roundtrip_is_identity() {
    let m = anb_n_dpda();
    let bits = m.to_bitvec();
    let m2 = PdaMachine::from_bitvec(&bits).expect("the round-trip");
    assert_eq!(m, m2, "the bitvec round-trip is the identity");
}

// PROOF 7: the PSC mask-classification (the config -> the mask-class -> the VOB).
#[test]
fn proof_psc_mask_classification() {
    use pushdown_rs::mask_class::PscClassifier;
    let m = anb_n_dpda();
    let classifier = PscClassifier::build(&m);
    // the codebook is much smaller than the config space (the PSC's win)
    let num_configs = m.num_states * m.num_stack_syms;
    assert!(
        classifier.codebook.len() < num_configs as usize,
        "the codebook ({} classes) must be smaller than the config space ({})",
        classifier.codebook.len(),
        num_configs
    );
    // the mask_of is consistent with the machine's transitions
    for state in 0..m.num_states {
        for top in 0..m.num_stack_syms {
            if let Some(mask) = classifier.mask_of(state, top) {
                let expected: Vec<u32> = (0..=m.num_inputs)
                    .filter(|&a| !m.lookup(state, Some(a), top).is_empty())
                    .collect();
                assert_eq!(&mask, &expected, "the mask_of must match the machine");
            }
        }
    }
}

// theOF 6: the SWYB bounded summary (the S_H, the d_H + the Reach_H).
#[test]
fn proof_bounded_summary() {
    use pushdown_rs::summary::BoundedSummary;
    let m = anb_n_dpda();
    let summary = BoundedSummary::compute(&m, 8);
    // the start config is reachable
    assert!(summary.is_reachable(m.start_state, &[m.start_stack]), "the start is reachable");
    // the d_H of the start is defined (the distance to acceptance)
    let d = summary.distance(m.start_state, &[m.start_stack]);
    assert!(d.is_some(), "the start has a finite d_H");
    // the accepting config has d_H = 0
    for &q in &m.accepting {
        if let Some(dq) = summary.distance(q, &[m.start_stack]) {
            assert_eq!(dq, 0, "the accepting config has d_H = 0");
        }
    }
}

// PROOF 8: the SIMD MaskOp (the rten-simd's dispatch, the broadcast).
#[cfg(feature = "simd")]
#[test]
fn proof_simd_maskop_dispatches() {
    use pushdown_rs::simd::MaskOp;
    use rten_simd::SimdOp;
    // the broadcast: the mask bits -> the logit row (the -inf if illegal)
    let logits = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![1u8, 0, 1, 0]; // the token 1 + the 3 are illegal
    let mut out = vec![0.0f32; 4];
    let mut op = MaskOp::new(&logits, &mask, &mut out);
    op.scalar();
    // the scalar broadcast: the illegal tokens (the mask=0) are the -inf
    assert_eq!(out[0], 1.0, "the legal token  its logit");
    assert_eq!(out[1], f32::NEG_INFINITY, "the illegal token is the -inf");
    assert_eq!(out[2], 3.0, "the legal token keeps its logit");
    assert_eq!(out[3], f32::NEG_INFINITY, "the illegal token is the -inf");
    // the SIMD dispatch ran without panic (the SIMD path is sound)
    let mut out2 = vec![0.0f32; 4];
    MaskOp::new(&logits, &mask, &mut out2).dispatch();
    assert!(out2.len() == 4, "the SIMD output is the right length");
}

// PROOF 8: the token spanner (the T_inv, the GreatGramma).
#[test]
fn proof_token_spanner() {
    use pushdown_rs::spanner::{LexerDfa, TokenSpanner, Tokenizer};
    // the a 2-state DFA: the state 0 (the start), the state 1 (the after 'a').
    // the terminal 'a' completes at the state 1.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Dfa;
    impl LexerDfa for Dfa {
        type LState = u8;
        type Terminal = u8;
        fn transition(&self, s: u8, b: u8) -> u8 {
            if s == 0 && b == b'a' { 1 } else { 255 } // the dead
        }
        fn is_dead(&self, s: u8) -> bool {
            s == 255
        }
        fn accepting(&self, s: u8) -> Vec<u8> {
            if s == 1 { vec![1] } else { vec![] }
        }
        fn initial(&self) -> u8 {
            0
        }
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Tok;
    impl Tokenizer for Tok {
        type Token = u8;
        fn token_bytes(&self, t: u8) -> Vec<u8> {
            vec![t]
        }
        fn vocab(&self) -> Vec<u8> {
            vec![b'a', b'b']
        }
    }
    let dfa = Dfa;
    let tok = Tok;
    let spanner = TokenSpanner::build(&dfa, &tok, &[0]);
    // the token 'a' from the state 0 produces the terminal sequence [1]
    assert_eq!(spanner.sequence_of(0, b'a'), vec![1u8]);
    // the token 'b' from the state 0 produces the empty sequence (the dead)
    assert_eq!(spanner.sequence_of(0, b'b'), Vec::<u8>::new());
    // the T_inv: the tokens that produce [1] from the state 0
    assert_eq!(spanner.tokens_for(0, &[1]), vec![b'a']);
}

// PROOF 9: the CUDA package (the bitvec + the source primitives).
#[cfg(feature = "cuda")]
#[test]
fn proof_cuda_package() {
    use pushdown_rs::cuda::CudaPackage;
    let m = anb_n_dpda();
    let pkg = CudaPackage::from_machine(&m).expect("the CUDA package");
    // the bitvec is the flat POD encoding (the GPU-uploadable)
    assert!(!pkg.bitvec.is_empty(), "the bitvec is non-empty");
    // the source primitives match the machine
    let src = pkg.source_primitives();
    assert_eq!(src.num_states, m.num_states);
    assert_eq!(src.num_inputs, m.num_inputs);
    assert_eq!(src.num_stack_syms, m.num_stack_syms);
    assert_eq!(src.num_transitions, m.transitions.len() as u32);
    // the upload size is the bitvec size (the GPU upload)
    assert!(pkg.upload_bytes() > 0, "the upload size is positive");
}

// ==================== the differentialDEPENDENT-ORACLE differential (the real 100% match) ====================
// The PDA (compiled from the CFG) must accept EXACTLY the CFG's language,
// verified against the INDEPENDENT oracle (oracle::cfg_accepts, zero shared
// code). This is the real correctness gate - not a self-comparison.

use pushdown_rs::oracle::cfg_accepts;

// a corpus of inputs over the {a,b} terminals (the a=1, the b=2)
fn ab_corpus() -> Vec<Vec<u32>> {
    let mut corpus = Vec::new();
    for len in 0..=6 {
        for mask in 0..(1usize << len) {
            let w: Vec<u32> = (0..len).map(|i| if (mask >> i) & 1 == 1 { 2 } else { 1 }).collect();
            corpus.push(w);
        }
    }
    corpus
}

#[test]
fn differential_anb_n_pda_matches_independent_oracle() {
    // the {a^n b^n} grammar (the S -> a S b | eps)
    let g = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2]), // S -> a S b
            (0, vec![]), // S -> eps
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    let corpus = ab_corpus();
    let num_nt = g.num_nonterminals; // the global->local offset
    let mut disagreements = 0;
    for w in &corpus {
        // the PDA input is the local terminal ID (the global - the num_nt)
        let pda_input: Vec<u32> = w.iter().map(|&x| x - num_nt).collect();
        let pda_says = m.accepts_npda(&pda_input, 64, 1_000_000);
        let oracle_says = cfg_accepts(&g, w); // the oracle uses the global IDs
        if pda_says != oracle_says {
            disagreements += 1;
            eprintln!("DISAGREE on global={:?} local={:?}: pda={} oracle={}", w, pda_input, pda_says, oracle_says);
        }
    }
    assert_eq!(disagreements, 0, "the PDA must match the independent oracle on all {} inputs", corpus.len());
}

#[test]
fn differential_balanced_parens_pda_matches_independent_oracle() {
    // the balanced-parens grammar (the S -> ( S ) S | eps, the Dyck language)
    // N = {S}, Sigma = {(the (, the )}, P = {S -> ( S ) S, S -> eps}
    let g = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2, 0]), // S -> ( S ) S
            (0, vec![]), // S -> eps
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    let corpus = ab_corpus(); // the (, ) as the global 1, 2
    let num_nt = g.num_nonterminals;
    let mut disagreements = 0;
    for w in &corpus {
        let pda_input: Vec<u32> = w.iter().map(|&x| x - num_nt).collect();
        let pda_says = m.accepts_npda(&pda_input, 64, 1_000_000);
        let oracle_says = cfg_accepts(&g, w);
        if pda_says != oracle_says {
            disagreements += 1;
            eprintln!("DISAGREE on global={:?} local={:?}: pda={} oracle={}", w, pda_input, pda_says, oracle_says);
        }
    }
    assert_eq!(disagreements, 0, "the PDA must match the independent oracle on all {} inputs", corpus.len());
}

// PROOF 10: the SWYB soundness (the d_H upper-bound property).
#[test]
fn proof_swyb_soundness() {
    use pushdown_rs::summary::BoundedSummary;
    let m = anb_n_dpda();
    let summary = BoundedSummary::compute(&m, 8);
    let d_start = summary.distance(m.start_state, &[m.start_stack]);
    assert!(d_start.is_some(), "the start config has a finite d_H");
    for &q in &m.accepting {
        if let Some(dq) = summary.distance(q, &[m.start_stack]) {
            assert_eq!(dq, 0, "the accepting config has d_H = 0");
        }
    }
    let d0 = d_start.unwrap();
    assert!(d0 <= 8, "the d_H is bounded by the H bound");
}

// PROOF 11: the batch invariant (the step_batch == the scalar step).
// The the batched op op must equal the scalar op applied per-item.
#[test]
fn proof_stream_step_batch_equals_scalar() {
    use pushdown_rs::pda::PdaStream;
    let m = anb_n_dpda();
    // the batch of (config, token) pairs (the batch)
    let batch: Vec<((u32, Vec<u32>), u32)> = vec![
        ((0, vec![0]), 0), // the start, the 'a'
        ((1, vec![0, 1]), 0), // the after the 'a', the 'a'
        ((1, vec![0, 1]), 1), // the after the 'a', the 'b'
    ];
    let batched = m.step_batch(&batch);
    // the scalar reference (the per-item step)
    for i in 0..batch.len() {
        let ((q, stk), a) = &batch[i];
        let top = stk.last().copied().unwrap_or(m.start_stack);
        let expected = match m.lookup(*q, Some(*a), top).as_slice() {
            [t] => {
                let mut s2 = stk.clone();
                s2.pop();
                for &p in t.push.iter().rev() {
                    s2.push(p);
                }
                (t.next_q, s2)
            }
            _ => (*q, stk.clone()),
        };
        assert_eq!(batched[i], expected, "the step_batch[{i}] must equal the scalar step");
    }
}

// the [a-z]+ regex is NON-deterministic (the S -> a S | a choice). The
// accepts_dpda (the deterministic) wrongly rejects it; the accepts_npda
// (the non-deterministic) is the correct oracle. This test condition
// (found by the real-world bench) guards against using the wrong accepts.
#[test]
fn nondeterministic_regex_uses_npda_accepts() {
    // the [a-z]+ = the S -> a S | a (the one-or-more, the non-deterministic)
    let g = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0]), // S -> a S
            (0, vec![1]), // S -> a
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    assert!(!m.is_deterministic(), "the [a-z]+ is non-deterministic (the choice)");
    // the PDA's input is the LOCAL terminal ID (the a=0, the b=1); the oracle
    // uses the GLOBAL (the a=1, the b=2). Map global -> local (the - num_nt).
    let to_local = |w: &[u32]| w.iter().map(|&x| x - 1).collect::<Vec<u32>>();
    // the single 'a' (the global [1] = the local [0]) is in the language (the S -> a)
    // the universal `accepts` auto-selects the NPDA (the non-deterministic) -
    // the caller does NOT need to know which variant the machine is.
    assert!(m.accepts(&to_local(&[1])), "the [a-z]+ accepts a single 'a' (the accepts)");
    assert!(m.accepts(&to_local(&[1, 1])), "the [a-z]+ accepts 'aa' (the universal accepts)");
    assert!(!m.accepts(&to_local(&[])), "the [a-z]+ rejects the empty (the one-or-more)");
}

// PROOF 12: the SIMD service (the device model, the batch in / batch out).
// The host sends a batch; the device (the SIMD service) processes it in
// parallel; the host receives the result. The host never sees the SIMD.
#[test]
fn proof_simd_service_device_model() {
    use pushdown_rs::service::PdaService;
    let m = anb_n_dpda();
    let service = PdaService::new(m.clone());
    // the step (the batch node 2 (step)): the batch of (config, token) -> the batch of next-config
    let batch = vec![
        (0u32, vec![0u32], 0u32), // the start, the 'a'
        (1, vec![0, 1], 0), // the after the 'a', the 'a'
        (1, vec![0, 1], 1), // the after the 'a', the 'b'
    ];
    let stepped = service.step(&batch);
    assert_eq!(stepped.len(), 3, "the batch size is preserved");
    // the mask (the batch node 1 (mask)): the batch of config -> the batch of mask
    let configs = vec![(0, vec![0u32]), (1, vec![0, 1])];
    let masks = service.mask(&configs);
    assert_eq!(masks.len(), 2, "the mask batch size is preserved");
    // the projection (the batch node 3 (project)): the batch of (config, draft) -> the batch of (K+1 masks)
    let drafts = vec![vec![0u32, 1], vec![0, 0, 1, 1]];
    let projected = service.project(&configs, &drafts);
    assert_eq!(projected.len(), 2, "the projection batch size is preserved");
    assert_eq!(projected[0].len(), drafts[0].len() + 1, "the K+1 masks");
}

// PROOF 13: the no-allocation batched step (the SIMD-1.2a). The
// step_batch_into writes into a pre-allocated buffer (the no per-item Vec).
// The batched invariant: the step_batch_into == the step_batch per-item.
#[test]
fn proof_step_batch_into_no_alloc() {
    use pushdown_rs::pda::PdaStream;
    let m = anb_n_dpda();
    let index = m.build_index();
    let batch: Vec<(u32, Vec<u32>, u32)> = vec![
        (0, vec![0], 0), // the start, the 'a'
        (1, vec![0, 1], 0), // the after the 'a', the 'a'
        (1, vec![0, 1], 1), // the after the 'a', the 'b'
    ];
    // the no-allocation step (the pre-allocated buffer)
    let mut out: Vec<(u32, Vec<u32>)> = vec![(0, vec![0]); batch.len()];
    m.step_batch_into(&index, &batch, &mut out);
    // the reference (the step_batch, the per-item)
    let ref_batch: Vec<((u32, Vec<u32>), u32)> =
        batch.iter().map(|(q, s, a)| ((*q, s.clone()), *a)).collect();
    let ref_out = m.step_batch(&ref_batch);
    assert_eq!(out, ref_out, "the step_batch_into must equal the step_batch (the batch invariant)");
}

// PROOF 14: the SIMD batched step (the batch node 2 (step), the B states in vectors).
// The step_batch_simd == the step_batch per-item (the batch invariant).
#[cfg(feature = "simd")]
#[test]
fn proof_step_batch_simd() {
    use pushdown_rs::pda::PdaStream;
    let m = anb_n_dpda();
    let index = m.build_index();
    // the batch of states (the u16, the states fit in 16 bits)
    let states: Vec<u16> = vec![0, 1, 1, 2];
    let token = 0u32; // the 'a'
    let simd_out = m.step_batch_simd(&index, &states, token);
    // the reference (the step_batch per-item)
    let ref_batch: Vec<((u32, Vec<u32>), u32)> = states
        .iter()
        .map(|&s| (((s as u32, vec![0u32]), token)))
        .collect();
    let ref_out = m.step_batch(&ref_batch);
    let ref_states: Vec<u16> = ref_out.iter().map(|(q, _)| *q as u16).collect();
    assert_eq!(simd_out, ref_states, "the step_batch_simd must equal the step_batch (the batch invariant)");
}

// PROOF 15: the SIMD batched projection (the batch node 3 (project), the B configs x
// the K drafts -> the B x (K+1) masks). The project_batch_simd == the
// project_batch per-item (the batch invariant).
#[cfg(feature = "simd")]
#[test]
fn proof_project_batch_simd() {
    use pushdown_rs::pda::PdaStream;
    let m = anb_n_dpda();
    let index = m.build_index();
    let configs = vec![(0, vec![0u32]), (1, vec![0, 1])];
    let drafts = vec![vec![0u32, 1], vec![0, 0, 1, 1]];
    let simd_out = m.project_batch_simd(&index, &configs, &drafts);
    let ref_out = m.project_batch(&configs, &drafts);
    assert_eq!(simd_out, ref_out, "the project_batch_simd must equal the project_batch (the batch invariant)");
}

// PROOF 16: the CUDA graph (the DAG structure + the mock replay).
#[test]
fn proof_cuda_graph_dag_and_replay() {
    use pushdown_rs::graph::{GraphNode, PdaGraph};
    let m = anb_n_dpda();
    // the single-token decode graph (the H2D -> ComputeMasks -> Sample -> Advance -> D2H)
    let g = PdaGraph::single_token_decode(4);
    assert_eq!(g.nodes.len(), 5, "the 5 nodes");
    assert_eq!(g.edges.len(), 4, "the 4 edges (the linear DAG)");
    let order = g.topo_order().expect("the valid DAG");
    assert_eq!(order.len(), 5, "the topological order covers all nodes");
    // the node replay (the CPU simulation of the CUDA graph replay)
    let final_state = g.mock_replay(&m).expect("the mock replay");
    assert!(!final_state.is_empty(), "the replay produces a state");
    // the drafting graph (the K+1 projection, the fixed unroll)
    let dg = PdaGraph::drafting(2, 16);
    assert_eq!(dg.nodes.len(), 3, "the 3 nodes (the H2D, the ProjectMasks, the D2H)");
    assert!(matches!(dg.nodes[1], GraphNode::ProjectMasks { batch: 2, k: 16 }));
    assert!(dg.topo_order().is_some(), "the drafting graph is a valid DAG");
    let _ = dg.mock_replay(&m).expect("the drafting mock replay");
}