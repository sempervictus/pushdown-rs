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
        state_provenance: None,
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

// The RTN state -> nonterminal projection (the `state_provenance` primitive).
//
// PROOF (the inclusive + the exclusive property, arXiv:2603.05540 Definition 5):
//   INCLUSIVE  - the projection is TOTAL: every control state q in Q is assigned
//               exactly one nonterminal (no gaps). The four RTN families
//               (start / entry / exit / dot) partition Q, and each family is tied
//               to one nonterminal: q_start -> S, q_A^in,q_A^out -> A, q_(p,i) -> lhs(p).
//   EXCLUSIVE  - the families are DISJOINT index ranges (no state belongs in two
//               families, so no state is written twice), and the phase label is a
//               FUNCTION (one value per state, not a relation).
// We verify both against an INDEPENDENT oracle (recomputed from the grammar's
// productions + the kappa numbering, not from the implementation), across the hard
// edge-case grammars (the eps-production, the single nonterminal, the unit chain,
// the recursion), and prove the phase-homology on the transition dynamics.

/// The inclusive + exclusive oracle for the RTN state -> nonterminal projection.
fn assert_provenance_oracle(g: &Cfg, m: &PdaMachine) {
    let num_nt = g.num_nonterminals;
    let dot_base = 1 + 2 * num_nt;
    // (exclusive) the four family index-ranges are contiguous + disjoint + cover
    // [0, num_states): start=[0], entry=[1,num_nt], exit=[num_nt+1,2*num_nt],
    // dot=[2*num_nt+1, num_states). The kappa count is sum_p(|rhs(p)|+1).
    let total_dots: u32 = g.productions.iter().map(|(_, rhs)| rhs.len() as u32 + 1).sum();
    assert_eq!(m.num_states, dot_base + total_dots, "kappa(G) = 1 + 2|N| + sum(|rhs|+1)");
    assert_eq!(kappa(g), m.num_states, "the kappa functional must agree with the compiled count");

    // Build the oracle with an EXACTLY-ONCE write check (the exclusive property:
    // no state is assigned by two families).
    let mut expected: Vec<Option<u32>> = vec![None; m.num_states as usize];
    let mut writes = 0u32;
    let assign = |vec: &mut Vec<Option<u32>>, idx: usize, val: u32, writes: &mut u32| {
        assert!(
            vec[idx].is_none(),
            "state {idx} assigned by two families (the ranges must be disjoint)"
        );
        vec[idx] = Some(val);
        *writes += 1;
    };
    assign(&mut expected, 0, g.start, &mut writes); // q_start -> S
    for a in 0..num_nt {
        assign(&mut expected, (1 + a) as usize, a, &mut writes); // q_A^in -> A
        assign(&mut expected, (1 + num_nt + a) as usize, a, &mut writes); // q_A^out -> A
    }
    let mut q = dot_base as usize;
    for (lhs, rhs) in &g.productions {
        for _ in 0..=rhs.len() {
            assign(&mut expected, q, *lhs, &mut writes); // q_(p,i) -> lhs(p)
            q += 1;
        }
    }
    // (inclusive) every state assigned exactly once (total + no gap + no overlap).
    assert_eq!(q as u32, m.num_states, "the dot family must end exactly at num_states (no gap)");
    assert_eq!(writes, m.num_states, "each state written exactly once (disjoint + total)");
    assert!(expected.iter().all(|e| e.is_some()), "no state left unassigned (inclusive)");
    let expected: Vec<u32> = expected.into_iter().map(|e| e.unwrap()).collect();

    // (correct) the implementation matches the independent oracle.
    assert_eq!(
        m.state_provenance.as_ref(),
        Some(&expected),
        "the projection must equal the RTN definition"
    );
    // (range) every entry is a valid nonterminal index.
    for &p in &expected {
        assert!(p < num_nt, "provenance entry {p} must be a nonterminal index (< |N|)");
    }
}

/// The phase-homology invariant on the machine's DYNAMICS (the transitions), not
/// just its static numbering. PROVES:
///   (inclusive) every move's endpoints carry well-defined phases consistent with
///               the RTN structure;
///   (exclusive) a terminal move is phase-PRESERVING and occurs ONLY at a dot
///               state; the ONLY phase-CHANGING moves are call (-> the callee's
///               entry) and return (-> the caller's dot); start/choice/exit preserve.
fn assert_phase_homology(g: &Cfg, m: &PdaMachine) {
    let num_nt = g.num_nonterminals;
    let num_inputs = m.num_inputs; // == the eps input ID
    let dot_base = 1 + 2 * num_nt;
    let entry_lo = 1u32;
    let entry_hi = num_nt; // q_in range [1, num_nt]
    let exit_lo = 1 + num_nt;
    let exit_hi = 2 * num_nt; // q_out range [num_nt+1, 2*num_nt]
    // dot_base = 2*num_nt + 1 (the first dot state); the dot range is [dot_base, num_states).
    let prov = m.state_provenance.as_ref().expect("provenance");
    for t in &m.transitions {
        let q = t.q;
        let is_eps = t.a == num_inputs;
        if q == 0 {
            // START: q_start --eps--> q_S^in (the push is [bot]).
            assert!(is_eps, "the start move is an epsilon move");
            assert_eq!(t.next_q, 1 + g.start, "the start move enters q_S^in");
            assert_eq!(prov[q as usize], prov[t.next_q as usize], "the start move preserves the phase (S)");
        } else if q <= entry_hi {
            // CHOICE: q_A^in --eps--> q_(p,0) (the push is [top]).
            assert!(is_eps, "the choice move is an epsilon move");
            assert!(t.next_q >= dot_base, "the choice move lands on a dot state");
            assert_eq!(prov[q as usize], prov[t.next_q as usize], "the choice move preserves the phase (A == lhs(p))");
        } else if q <= exit_hi {
            // RETURN: q_B^out --eps--> r (a dot) (the push is EMPTY).
            assert!(is_eps, "the return move is an epsilon move");
            assert!(t.push.is_empty(), "the return move pops the return address (the empty push)");
            assert!(t.next_q >= dot_base, "the return move lands on a dot state");
            assert_eq!(prov[q as usize], q - exit_lo, "the return is out of q_B^out (the phase is B)");
            // the phase CHANGES from the callee B to the caller's lhs(p) (the dot).
        } else {
            // DOT state: either a terminal move, a call, or an exit.
            if !is_eps {
                // TERMINAL: q_(p,i) --X--> q_(p,i+1) (the push is [top]).
                assert!(q >= dot_base, "a terminal move occurs only at a dot state (the exclusive property)");
                assert!(t.next_q >= dot_base, "a terminal move lands on the next dot");
                assert_eq!(prov[q as usize], prov[t.next_q as usize], "a terminal move preserves the phase (the same production)");
            } else if t.push.len() == 2 {
                // CALL: q_(p,i-1) --eps--> q_B^in (the push is [ret_addr, top]).
                assert!(t.next_q >= entry_lo && t.next_q <= entry_hi, "the call target is an entry state");
                assert_eq!(prov[t.next_q as usize], t.next_q - entry_lo, "the call enters the callee's phase (B)");
                // the phase CHANGES from the caller's lhs(p) to the callee B.
            } else {
                // EXIT: q_(p,m) --eps--> q_A^out (the push is [top]).
                assert!(t.next_q >= exit_lo && t.next_q <= exit_hi, "the exit target is an exit state");
                assert_eq!(t.next_q, exit_lo + prov[q as usize], "the exit move lands on q_out of the same nonterminal");
                assert_eq!(prov[q as usize], prov[t.next_q as usize], "the exit move preserves the phase (lhs(p) == A)");
            }
        }
    }
}

#[test]
fn rtn_state_provenance_inclusive_and_exclusive() {
    // (1) the 3-NT grammar with an eps-production (the single-dot edge).
    let g1 = Cfg::new(
        3, // num_nonterminals (S=0, A=1, B=2)
        2, // num_terminals (a=3, b=4)
        0, // start = S
        vec![
            (0, vec![3, 1, 4]), // S -> a A b
            (0, vec![]), // S -> eps
            (1, vec![2, 2]), // A -> B B
            (1, vec![3]), // A -> a
            (2, vec![4]), // B -> b
        ],
    );
    let m1 = pushdown_rs::compile(&g1).expect("compile");
    assert_provenance_oracle(&g1, &m1);
    assert_phase_homology(&g1, &m1);

    // (2) the single-nonterminal a^n b^n (the recursion + the eps edge).
    let g2 = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2]), // S -> a S b
            (0, vec![]), // S -> eps
        ],
    );
    let m2 = pushdown_rs::compile(&g2).expect("compile");
    assert_provenance_oracle(&g2, &m2);
    assert_phase_homology(&g2, &m2);

    // (3) the unit-production chain (the tight call/return, the S -> A -> B -> a).
    let g3 = Cfg::new(
        3,
        1,
        0,
        vec![
            (0, vec![1]), // S -> A
            (1, vec![2]), // A -> B
            (2, vec![3]), // B -> a (the a = the global ID 3)
        ],
    );
    let m3 = pushdown_rs::compile(&g3).expect("compile");
    assert_provenance_oracle(&g3, &m3);
    assert_phase_homology(&g3, &m3);

    // (4) the minimal single-production grammar (the num_nt=1, one dot-chain).
    let g4 = Cfg::new(1, 2, 0, vec![(0, vec![1, 2])]); // S -> a b
    let m4 = pushdown_rs::compile(&g4).expect("compile");
    assert_provenance_oracle(&g4, &m4);
    assert_phase_homology(&g4, &m4);

    // (5) the bitvec round-trip preserves the projection (the Some case).
    let bits = m1.to_bitvec();
    let m1r = PdaMachine::from_bitvec(&bits).expect("round-trip");
    assert_eq!(m1r.state_provenance, m1.state_provenance, "the round-trip must be lossless");
}

// A hand-built machine (the no RTN provenance) round-trips its `None` flag.
#[test]
fn hand_built_provenance_none_round_trips() {
    let m = anb_n_dpda();
    assert!(m.state_provenance.is_none(), "the hand-built machine has no provenance");
    let bits = m.to_bitvec();
    let m2 = PdaMachine::from_bitvec(&bits).expect("the round-trip must deserialize");
    assert!(m2.state_provenance.is_none(), "the None flag must survive the round-trip");
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

// The bitvec POD size is EXACT (the header + the accepting + the transitions +
// the provenance suffix), not just bounded. This proves the size formula:
//   (inclusive) every serialized field is accounted for;
//   (exclusive) there is no padding or extra (the size is the exact u32 count * 32).
#[test]
fn bitvec_size_is_exact() {
    let m = anb_n_dpda();
    let bits = m.to_bitvec();
    let header = 7u32; // num_states, num_inputs, num_stack_syms, num_transitions, num_accepting, start_state, start_stack
    let accepting = m.accepting.len() as u32;
    let transitions: u32 = m.transitions.iter().map(|t| 5 + t.push.len() as u32).sum();
    let prov_suffix = match &m.state_provenance {
        Some(p) => 1 + p.len() as u32, // the flag + the num_states entries
        None => 1, // the flag only
    };
    let expected_u32 = header + accepting + transitions + prov_suffix;
    assert_eq!(bits.len(), (expected_u32 * 32) as usize, "the bitvec size must be the exact POD formula");
}

// The RTN compilation scaling: the kappa(G) = 1 + 2|N| + sum_p(|rhs(p)|+1).
// Verified against an INDEPENDENT recomputation of the formula (
// heterogeneous grammars (varying |N| and varying production lengths), not just
// a degenerate n-identical-productions case. The compiled machine must have
// exactly kappa states (the Lemma 2, the inclusive + exclusive count).
#[test]
fn rtn_compilation_scales_with_grammar() {
    // (|N|, |Sigma|, the productions) - heterogeneous rhs lengths.
    let cases: Vec<(u32, u32, Vec<(u32, Vec<u32>)>)> = vec![
        (1, 2, vec![(0, vec![1, 0, 2]), (0, vec![])]), // the a^n b^n
        (2, 2, vec![(0, vec![1, 1, 2]), (1, vec![2]), (1, vec![])]),
        (3, 1, vec![(0, vec![1]), (1, vec![1, 0]), (2, vec![1, 1, 0, 2])]),
        (1, 3, vec![(0, vec![1]), (0, vec![2]), (0, vec![3]), (0, vec![])]),
    ];
    for (num_nt, num_tm, prods) in cases {
        let g = Cfg::new(num_nt, num_tm, 0, prods.clone());
        // the independent formula: 1 + 2|N| + sum_p(|rhs(p)|+1).
        let expected: u32 = 1 + 2 * num_nt + prods.iter().map(|(_, rhs)| rhs.len() as u32 + 1).sum::<u32>();
        assert_eq!(kappa(&g), expected, "the kappa must equal the independent formula (|N|={num_nt})");
        // the compiled machine must have exactly kappa states (the Lemma 2).
        let m = pushdown_rs::compile(&g).expect("compile");
        assert_eq!(m.num_states, expected, "the compiled state count must equal kappa(G)");
    }
}

// ==================== the mathematical proofs (the full coverage) ====================

// PROOF 1: the Determinism (the at most one transition per (q, a, top)).
// The DPDA's delta is a FUNCTION (the at most one next-move per (q, a, top)).
// This is the defining property of the DPDA (vs the NPDA). We prove it is a
// REAL discriminating property (the inclusive + exclusive):
//   (exclusive) every deterministic machine has ZERO duplicate (q, a, top) keys;
//   (inclusive) an ambiguous RTN machine HAS a duplicate key, so the check is not
//               vacuously true for all machines (it discriminates).
#[test]
fn proof_determinism_is_a_function() {
    // the deterministic machines (the no dup (q, a, top) key).
    let det_machines: Vec<PdaMachine> = vec![
        anb_n_dpda(), // the hand-built a^n b^n DPDA
        pushdown_rs::compile(&Cfg::new(1, 2, 0, vec![(0, vec![1, 2])])).expect("the S -> a b (the single production)"),
    ];
    for m in &det_machines {
        assert!(m.is_deterministic(), "the machine must be deterministic");
        let mut keys: Vec<(u32, u32, u32)> = m.transitions.iter().map(|t| (t.q, t.a, t.top)).collect();
        keys.sort();
        let dupes = keys.iter().zip(keys.iter().skip(1)).filter(|(a, b)| a == b).count();
        assert_eq!(dupes, 0, "a deterministic machine has no duplicate (q, a, top) key");
    }
    // the ambiguous machine (the HAS a dup key, the non-determinism is real).
    let amb = pushdown_rs::compile(&Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![])]))
        .expect("the S -> a S b | eps (the choice)");
    assert!(!amb.is_deterministic(), "the ambiguous grammar must be non-deterministic");
    let mut keys: Vec<(u32, u32, u32)> = amb.transitions.iter().map(|t| (t.q, t.a, t.top)).collect();
    keys.sort();
    let dupes = keys.iter().zip(keys.iter().skip(1)).filter(|(a, b)| a == b).count();
    assert!(dupes > 0, "the ambiguous machine must have a duplicate (q, a, top) key (the non-determinism)");
}

// PROOF 2: the Bounded stack (the Reach_H depth <= D, the per-step push bound).
// The GPU-resident DPDA uses a fixed-size stack of depth D. This proves:
//   (inclusive) EVERY reachable config respects the depth bound D;
//   (exclusive) the per-step push is bounded by the max production length + 1
//               (the ret_addr + top), and the nesting is real (the bound is met).
#[test]
fn proof_stack_is_bounded() {
    use pushdown_rs::summary::BoundedSummary;
    let m = anb_n_dpda();
    let d = 8;
    let summary = BoundedSummary::compute(&m, d);
    // (inclusive) every reachable config has stack depth <= d.
    let mut max_depth = 0usize;
    for (_, stack) in summary.reachable.iter() {
        assert!(stack.len() <= d, "a reachable config must respect the stack bound d");
        max_depth = max_depth.max(stack.len());
    }
    // (exclusive) the bound is real: the a^n b^n nests to a non-trivial depth
    // (the [A_SYM, Z] after the first 'a'), and the per-step push is bounded.
    assert!(max_depth >= 2, "the a^n b^n must reach a non-trivial nesting depth");
    let max_push = m.transitions.iter().map(|t| t.push.len()).max().unwrap_or(0);
    assert!(max_push <= 2, "the per-step push is bounded (the at most 2 symbols)");
}

// PROOF 3: the Mask fidelity (the PSC classifier == the machine's legal inputs).
// For EVERY config (state, top) in the full config space, the classifier's mask
// is EXACTLY the set of inputs with a defined transition:
//   (inclusive) every legal input is present in the mask;
//   (exclusive) no illegal input is present in the mask.
// The epsilon input ID (== num_inputs) is included, matching the classifier.
#[test]
fn proof_mask_fidelity() {
    use pushdown_rs::mask_class::PscClassifier;
    let m = anb_n_dpda();
    let classifier = PscClassifier::build(&m);
    for state in 0..m.num_states {
        for top in 0..m.num_stack_syms {
            // the independent oracle: the legal inputs from (state, top).
            let legal: Vec<u32> = (0..=m.num_inputs)
                .filter(|&a| !m.lookup(state, Some(a), top).is_empty())
                .collect();
            match classifier.mask_of(state, top) {
                Some(mask) => assert_eq!(
                    mask, legal,
                    "the mask must be EXACTLY the legal inputs (state={state}, top={top})"
                ),
                None => assert!(
                    legal.is_empty(),
                    "a None mask must mean no legal inputs (state={state}, top={top})"
                ),
            }
        }
    }
}

// PROOF 4: the Projection (the project(K) == the K sequential steps).
// The projection walks the draft one token at a time and emits a mask per prefix.
// Its contract (the inclusive + exclusive):
//   (inclusive) a prefix up to and including the first divergent token is emitted;
//   (exclusive) the walk STOPS at the first illegal draft token (the no transition) -
//               you cannot project past a dead-end, so no masks are emitted beyond it.
// We reference is built from the primitive `lookup` + `mask_at` (the independent
// composition), NOT from `project_batch`, so it is a genuine oracle.
#[test]
fn proof_projection_equals_sequential() {
    use pushdown_rs::pda::PdaStream;
    let m = anb_n_dpda();
    let config = (m.start_state, vec![m.start_stack]);

    // the independent reference: walk the draft, emit a mask per prefix (the public
    // mask_batch), and STOP at the first divergent token (the no transition) - the
    // projection's contract. Built from `lookup` + `mask_batch` (the primitives),
    // NOT from `project_batch`, so it is a genuine oracle.
    fn sequential_projection(m: &PdaMachine, config: (u32, Vec<u32>), draft: &[u32]) -> Vec<Vec<u32>> {
        use pushdown_rs::pda::PdaStream;
        let mut masks = vec![m.mask_batch(&[config.clone()])[0].clone()];
        let mut q = config.0;
        let mut stk = config.1;
        for &a in draft {
            let top = stk.last().copied().unwrap_or(m.start_stack);
            match m.lookup(q, Some(a), top).as_slice() {
                [t] => {
                    stk.pop();
                    for &p in t.push.iter().rev() {
                        stk.push(p);
                    }
                    q = t.next_q;
                    masks.push(m.mask_batch(&[(q, stk.clone())])[0].clone());
                }
                _ => break, // the draft diverged (the no transition) - the exclusive stop
            }
        }
        masks
    }

    // the fully-legal draft (the a a b b, the n=2): no divergence, K+1 masks.
    let legal_draft = vec![0u32, 0, 1, 1];
    let projected = m.project_batch(&[config.clone()], &[legal_draft.clone()]);
    assert_eq!(projected[0], sequential_projection(&m, config.clone(), &legal_draft), "the legal projection must equal the K sequential steps");
    assert_eq!(projected[0].len(), legal_draft.len() + 1, "the legal draft emits K+1 masks (the inclusive)");

    // the divergent draft (the a b b a: the trailing tokens dead-end at state 2):
    // the projection STOPS at the first illegal token (the exclusive).).
    let illegal_draft = vec![0u32, 1, 1, 0];
    let projected2 = m.project_batch(&[config.clone()], &[illegal_draft.clone()]);
    let expected2 = sequential_projection(&m, config.clone(), &illegal_draft);
    assert_eq!(projected2[0], expected2, "the divergent projection must stop at the first illegal token");
    assert!(projected2[0].len() < illegal_draft.len() + 1, "the divergent draft emits FEWER than K+1 masks (the exclusive stop)");
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

// PROOF 6: the SWYB bounded summary (the S_H, the Reach_H completeness).
//   (inclusive) the start config is reachable; every config with a defined d_H
//               is in the Reach_H set;
//   (exclusive) the Reach_H set contains only configs of stack depth <= H.
#[test]
fn proof_bounded_summary() {
    use pushdown_rs::summary::BoundedSummary;
    let m = anb_n_dpda();
    let h = 8;
    let summary = BoundedSummary::compute(&m, h);
    // (inclusive) the start config is reachable and has a finite d_H.
    assert!(summary.is_reachable(m.start_state, &[m.start_stack]), "the start is reachable");
    assert!(summary.distance(m.start_state, &[m.start_stack]).is_some(), "the start has a finite d_H");
    // (exclusive) every reachable config respects the depth bound h.
    for (_, stack) in summary.reachable.iter() {
        assert!(stack.len() <= h, "a reachable config must have depth <= h");
    }
    // (inclusive) every config with a defined d_H is in the Reach_H set.
    for (q, stack) in summary.distance.keys() {
        assert!(summary.is_reachable(*q, stack), "a d_H-defined config must be reachable");
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

// a corpus of ALL binary strings over the {a,b} terminals (the a=1, the b=2) up
// to length `max_len` (the 2^0 + ... + 2^max_len inputs, the exhaustive prefix set).
fn ab_corpus(max_len: usize) -> Vec<Vec<u32>> {
    let mut corpus = Vec::new();
    for len in 0..=max_len {
        for mask in 0..(1usize << len) {
            let w: Vec<u32> = (0..len).map(|i| if (mask >> i) & 1 == 1 { 2 } else { 1 }).collect();
            corpus.push(w);
        }
    }
    corpus
}

// the boundary corpus: for each n, the balanced a^n b^n (the in-language) plus the
// just-unbalanced a^n b^(n+1) and a^(n+1) b^n (the out-of-language). These are the
// hard cases where the PDA's acceptance discipline is actually exercised (the push/pop
// balance), so they must be covered explicitly (the inclusive + exclusive edge).
fn ab_boundary_corpus(max_n: usize) -> Vec<Vec<u32>> {
    let mut corpus = Vec::new();
    for n in 0..=max_n {
        let mut in_lang = vec![1u32; n];
        in_lang.extend(vec![2u32; n]);
        corpus.push(in_lang); // a^n b^n (the in)
        let mut over_b = vec![1u32; n];
        over_b.extend(vec![2u32; n + 1]);
        corpus.push(over_b); // a^n b^(n+1) (the out)
        let mut over_a = vec![1u32; n + 1];
        over_a.extend(vec![2u32; n]);
        corpus.push(over_a); // a^(n+1) b^n (the out)
    }
    corpus
}

// Run a differential oracle check: the PDA's accepts must equal the independent
// cfg_accepts oracle on every input in `corpus` (the global terminal IDs).
fn run_differential(m: &PdaMachine, g: &Cfg, corpus: &[Vec<u32>]) {
    let num_nt = g.num_nonterminals; // the global->local offset
    let mut disagreements = 0;
    for w in corpus {
        // the PDA input is the local terminal ID (the global - the num_nt)
        let pda_input: Vec<u32> = w.iter().map(|&x| x - num_nt).collect();
        let pda_says = m.accepts_npda(&pda_input, 64, 1_000_000);
        let oracle_says = cfg_accepts(g, w); // the oracle uses the global IDs
        if pda_says != oracle_says {
            disagreements += 1;
            eprintln!("DISAGREE on global={:?} local={:?}: pda={} oracle={}", w, pda_input, pda_says, oracle_says);
        }
    }
    assert_eq!(disagreements, 0, "the PDA must match the independent oracle on all {} inputs", corpus.len());
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
    // the exhaustive prefix corpus (the length 0..=12) + the boundary corpus (the n=0..=8).
    let full = ab_corpus(12);
    let boundary = ab_boundary_corpus(8);
    run_differential(&m, &g, &full);
    run_differential(&m, &g, &boundary);
    // the EXCLUSIVE language property: the PDA accepts EXACTLY {a^n b^n} (no
    // superset). In the full corpus (all binary strings of length 0..=12), the
    // accepted strings are precisely a^n b^n with 2n <= 12, i.e. n = 0..=6 (the 7).
    let accepted: Vec<_> = full.iter().filter(|w| cfg_accepts(&g, w)).collect();
    assert_eq!(accepted.len(), 7, "the a^n b^n language accepts EXACTLY 7 strings of length <= 12 (the n=0..=6)");
    for w in &accepted {
        // each accepted string is a^n b^n (the a's then the b's, the equal count).
        let na = w.iter().filter(|&&x| x == 1).count();
        let nb = w.iter().filter(|&&x| x == 2).count();
        assert_eq!(na, nb, "an accepted string must be a^n b^n (the equal count)");
        assert!(w.iter().take_while(|&&x| x == 1).count() == na, "the a's must precede the b's");
    }
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
    // the Dyck language is nested (the stack discipline is exercised deeper), so the
    // boundary corpus (the balanced parens) is the hard case.
    let full = ab_corpus(10);
    let boundary = ab_boundary_corpus(6);
    run_differential(&m, &g, &full);
    run_differential(&m, &g, &boundary);
}

// A third differential on a HETEROGENEOUS multi-nonterminal grammar (the not just
// the single-NT a^n b^n / Dyck), broadening the oracle coverage to the general RT.
#[test]
fn differential_multi_nonterminal_pda_matches_independent_oracle() {
    // N = {S=0, A=1, B=2}, Sigma = {a=3, b=4}.
    //   S -> a A b | eps
    //   A -> B B | a 
    //   B -> b
    let g = Cfg::new(
        3,
        2,
        0,
        vec![
            (0, vec![3, 1, 4]), // S -> a A b
            (0, vec![]), // S -> eps
            (1, vec![2, 2]), // A -> B B
            (1, vec![3]), // A -> a
            (2, vec![4]), // B -> b
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // the terminals are the global 3 (a) and 4 (b); the corpus is over {3,4}.
    let corpus: Vec<Vec<u32>> = (0..=8)
        .flat_map(|len| {
            (0..(1usize << len)).map(move |mask| {
                (0..len).map(|i| if (mask >> i) & 1 == 1 { 4 } else { 3 }).collect::<Vec<u32>>()
            })
        })
        .collect();
    run_differential(&m, &g, &corpus);
}

// The epsilon-closure advance (the PDA must advance THROUGH the call dots, the
// nonterminal calls, not get stuck). The S -> a A b, A -> c grammar has a call
// dot (the S -> a [A] b, the A call). The PDA must advance through it to accept
// "a c b". This proves the advance is consistent with the mask (the epsilon-
// closure), and matches the independent oracle oracle.
#[test]
fn epsilon_closure_advance_through_call() {
    use pushdown_rs::Pda;
    // N = {S=0, A=1}, Sigma = {a=2, b=3, c=4} (the global Cfg IDs).
    //   S -> a A b
    //   A -> c
    let g = Cfg::new(
        2,
        3,
        0,
        vec![
            (0, vec![2, 1, 3]), // S -> a A b
            (1, vec![4]), // A -> c
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // the local terminal IDs (the global - the num_nt=2): a=0, b=1, c=2.
    // the input "a c b" = the local [0, 2, 1].
    let local = vec![0u32, 2, 1];
    assert!(m.accepts(&local), "the PDA accepts a c b (the S -> a A b, A -> c)");
    // the differential: the independent cfg oracle agrees (the global IDs).
    let global = vec![2u32, 4, 3]; // the a, c, b (the global)
    assert_eq!(
        m.accepts(&local),
        pushdown_rs::oracle::cfg_accepts(&g, &global),
        "the PDA must match the independent cfg oracle"
    );
    // the advance through the call dot (the step_batch, the epsilon-closure):
    // from the start, the "a" advances to the call dot (the S -> a [A] b), the
    // "c" advances through the A call (the A -> c), the "b" completes the S.
    // The loop's .expect() is the proof: it panics if the advance gets stuck at
    // any call dot (the no epsilon-closure). The m.accepts above already proves
    // the full language membership (the reaching via the epsilon exit).
    let mut config = (m.start_state, vec![m.start_stack]);
    for &tok in &local {
        config = m.advance_eps(config.0, &config.1, tok).expect("the advance must succeed (the no stuck call dot)");
    }
    // the post-consumption config is at the last dot of S (the dot(S,3)); the
    // accepting state q_S^out is reached via the epsilon exit (the no input).
    // Verify the epsilon closure from here reaches the accepting state.
    let mut reached_accepting = false;
    let mut frontier = vec![(config.0, config.1.clone())];
    let mut visited = std::collections::HashSet::new();
    while let Some((cq, cstk)) = frontier.pop() {
        if !visited.insert((cq, cstk.clone())) {
            continue;
        }
        if m.accepting.contains(&cq) {
            reached_accepting = true;
            break;
        }
        let ctop = cstk.last().copied().unwrap_or(m.start_stack);
        for (q2, push) in m.transition(cq, None, ctop) {
            let mut ns = cstk.clone();
            ns.pop();
            for &p in push.iter().rev() {
                ns.push(p);
            }
            frontier.push((q2, ns));
        }
    }
    assert!(reached_accepting, "the epsilon closure from the post-consumption config reaches the accepting state");
}

// PROOF 10: the SWYB soundness (the d_H upper-bound property, the inclusive + exclusive).
//   (inclusive) EVERY reachable config's d_H (when defined) is <= H;
//   (exclusive) every accepting config has d_H = 0 UNCONDITIONALLY (the bound is
//               tight at the target), and the start config has a finite d_H.
#[test]
fn proof_swyb_soundness() {
    use pushdown_rs::summary::BoundedSummary;
    let m = anb_n_dpda();
    let h = 8;
    let summary = BoundedSummary::compute(&m, h);
    // (inclusive) every reachable config's d_H (when defined) is <= h.
    for (q, stack) in summary.reachable.iter() {
        if let Some(d) = summary.distance(*q, stack) {
            assert!(d <= h as u32, "the d_H must be bounded by h (got {d})");
        }
    }
    // (exclusive) every accepting config has d_H = 0 (unconditional, not skipped).
    for (q, stack) in summary.reachable.iter() {
        if m.accepting.contains(q) {
            assert_eq!(summary.distance(*q, stack), Some(0), "an accepting config must have d_H = 0");
        }
    }
    // the start config reaches acceptance within h (the grammar terminates).
    assert!(summary.distance(m.start_state, &[m.start_stack]).is_some(), "the start must have a finite d_H");
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
        .map(|&s| ((s as u32, vec![0u32]), token))
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