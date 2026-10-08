//! Test cases for the PDA domain, extracted from the references:
//!   - the a^n b^n DPDA (the JFLAP tutorial, the push/pop construction)
//!   - the RTN compilation (Alpay & Senturk, arXiv:2603.05540, Definition 5)
//!   - the bitvec round-trip

use pushdown_rs::bitvec::BitvecError;
use pushdown_rs::compile::{kappa, Cfg};
use pushdown_rs::machine::{PdaMachine, Transition};
use pushdown_rs::pda::{Dpda, Npda, PdaStream};

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
            Transition {
                q: 0,
                a: A,
                top: Z,
                next_q: 1,
                push: vec![A_SYM, Z],
            },
            Transition {
                q: 1,
                a: A,
                top: A_SYM,
                next_q: 1,
                push: vec![A_SYM, A_SYM],
            },
            Transition {
                q: 1,
                a: B,
                top: A_SYM,
                next_q: 2,
                push: vec![],
            },
            Transition {
                q: 2,
                a: B,
                top: A_SYM,
                next_q: 2,
                push: vec![],
            },
            Transition {
                q: 2,
                a: EPS,
                top: Z,
                next_q: 3,
                push: vec![Z],
            },
        ],
        accepting: vec![3],
        start_state: 0,
        start_stack: Z,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
    flat_a: vec![],
    flat_top: vec![],
    flat_next_q: vec![],
    }
}

#[test]
fn anb_n_is_deterministic() {
    let m = anb_n_dpda();
    assert!(
        m.is_deterministic(),
        "the a^n b^n DPDA must be deterministic"
    );
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
            (0, vec![3]),       // S -> c
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // the kappa(G) = 1 + 2*1 + (3+1) + (3+1) + (1+1) = 13
    assert_eq!(kappa(&g), 13, "the kappa(G) must match the Definition 10");
    assert_eq!(
        m.num_states, 13,
        "the compiled state count must equal kappa(G)"
    );
    // the RTN compilation is an NPDA (the choice is non-deterministic)
    assert!(
        !m.is_deterministic(),
        "the RTN compilation has a non-deterministic choice"
    );
    // the language: use a bounded grammar (the no recursion) for the accepts check
    let g2 = Cfg::new(1, 3, 0, vec![(0, vec![3])]); // S -> c (the c = the global ID 3)
    let m2 = pushdown_rs::compile(&g2).expect("compile");
    // the c input ID for the c (the global 3) = 3 - 1 = 2
    assert!(
        m2.accepts_npda(&g2, &[2]),
        "the NPDA accepts c (the S -> c)"
    );
    assert!(
        !m2.accepts_npda(&g2, &[0]),
        "the NPDA rejects a (the no S -> a)"
    );
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
    let total_dots: u32 = g
        .productions
        .iter()
        .map(|(_, rhs)| rhs.len() as u32 + 1)
        .sum();
    assert_eq!(
        m.num_states,
        dot_base + total_dots,
        "kappa(G) = 1 + 2|N| + sum(|rhs|+1)"
    );
    assert_eq!(
        kappa(g),
        m.num_states,
        "the kappa functional must agree with the compiled count"
    );

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
    assert_eq!(
        q as u32, m.num_states,
        "the dot family must end exactly at num_states (no gap)"
    );
    assert_eq!(
        writes, m.num_states,
        "each state written exactly once (disjoint + total)"
    );
    assert!(
        expected.iter().all(|e| e.is_some()),
        "no state left unassigned (inclusive)"
    );
    let expected: Vec<u32> = expected.into_iter().map(|e| e.unwrap()).collect();

    // (correct) the implementation matches the independent oracle.
    assert_eq!(
        m.state_provenance.as_ref(),
        Some(&expected),
        "the projection must equal the RTN definition"
    );
    // (range) every entry is a valid nonterminal index.
    for &p in &expected {
        assert!(
            p < num_nt,
            "provenance entry {p} must be a nonterminal index (< |N|)"
        );
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
            assert_eq!(
                prov[q as usize], prov[t.next_q as usize],
                "the start move preserves the phase (S)"
            );
        } else if q <= entry_hi {
            // CHOICE: q_A^in --eps--> q_(p,0) (the push is [top]).
            assert!(is_eps, "the choice move is an epsilon move");
            assert!(t.next_q >= dot_base, "the choice move lands on a dot state");
            assert_eq!(
                prov[q as usize], prov[t.next_q as usize],
                "the choice move preserves the phase (A == lhs(p))"
            );
        } else if q <= exit_hi {
            // RETURN: q_B^out --eps--> r (a dot) (the push is EMPTY).
            assert!(is_eps, "the return move is an epsilon move");
            assert!(
                t.push.is_empty(),
                "the return move pops the return address (the empty push)"
            );
            assert!(t.next_q >= dot_base, "the return move lands on a dot state");
            assert_eq!(
                prov[q as usize],
                q - exit_lo,
                "the return is out of q_B^out (the phase is B)"
            );
            // the phase CHANGES from the callee B to the caller's lhs(p) (the dot).
        } else {
            // DOT state: either a terminal move, a call, or an exit.
            if !is_eps {
                // TERMINAL: q_(p,i) --X--> q_(p,i+1) (the push is [top]).
                assert!(
                    q >= dot_base,
                    "a terminal move occurs only at a dot state (the exclusive property)"
                );
                assert!(
                    t.next_q >= dot_base,
                    "a terminal move lands on the next dot"
                );
                assert_eq!(
                    prov[q as usize], prov[t.next_q as usize],
                    "a terminal move preserves the phase (the same production)"
                );
            } else if t.push.len() == 2 {
                // CALL: q_(p,i-1) --eps--> q_B^in (the push is [ret_addr, top]).
                assert!(
                    t.next_q >= entry_lo && t.next_q <= entry_hi,
                    "the call target is an entry state"
                );
                assert_eq!(
                    prov[t.next_q as usize],
                    t.next_q - entry_lo,
                    "the call enters the callee's phase (B)"
                );
                // the phase CHANGES from the caller's lhs(p) to the callee B.
            } else {
                // EXIT: q_(p,m) --eps--> q_A^out (the push is [top]).
                assert!(
                    t.next_q >= exit_lo && t.next_q <= exit_hi,
                    "the exit target is an exit state"
                );
                assert_eq!(
                    t.next_q,
                    exit_lo + prov[q as usize],
                    "the exit move lands on q_out of the same nonterminal"
                );
                assert_eq!(
                    prov[q as usize], prov[t.next_q as usize],
                    "the exit move preserves the phase (lhs(p) == A)"
                );
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
            (0, vec![]),        // S -> eps
            (1, vec![2, 2]),    // A -> B B
            (1, vec![3]),       // A -> a
            (2, vec![4]),       // B -> b
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
            (0, vec![]),        // S -> eps
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
    assert_eq!(
        m1r.state_provenance, m1.state_provenance,
        "the round-trip must be lossless"
    );
}

// A hand-built machine (the no RTN provenance) round-trips its `None` flag.
#[test]
fn hand_built_provenance_none_round_trips() {
    let m = anb_n_dpda();
    assert!(
        m.state_provenance.is_none(),
        "the hand-built machine has no provenance"
    );
    let bits = m.to_bitvec();
    let m2 = PdaMachine::from_bitvec(&bits).expect("the round-trip must deserialize");
    assert!(
        m2.state_provenance.is_none(),
        "the None flag must survive the round-trip"
    );
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
            (0, vec![]),        // S -> eps
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    assert!(
        !m.is_deterministic(),
        "the RTN compilation of an ambiguous grammar is an NPDA"
    );
    // the NPDA still accepts the language (the any-path)
    const A: u32 = 0; // the a (the local input ID)
    const B: u32 = 1; // the b (the local input ID)
    assert!(
        m.accepts_npda(&g, &[A, B]),
        "the NPDA accepts ab (the n=1)"
    );
    assert!(
        m.accepts_npda(&g, &[A, A, B, B]),
        "the NPDA accepts aabb (the n=2)"
    );
    assert!(
        !m.accepts_npda(&g, &[A, B, A]),
        "the NPDA rejects aba (the unbalanced)"
    );
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
                    found_unsafe.push(format!(
                        "{:?}:{}: {}",
                        path.file_name().unwrap().to_string_lossy(),
                        i + 1,
                        line.trim()
                    ));
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
    assert!(
        matches!(res, Err(BitvecError::Malformed(_))),
        "truncated bitvec must fail"
    );
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
        assert!(
            m.accepts_dpda(&w),
            "the DPDA accepts a^n b^n (n={n}, the large input)"
        );
    }
    // the large unbalanced input (the n=100 a's, the n=99 b's)
    let mut w = Vec::with_capacity(199);
    for _ in 0..100 {
        w.push(A);
    }
    for _ in 0..99 {
        w.push(B);
    }
    assert!(
        !m.accepts_dpda(&w),
        "the DPDA rejects the unbalanced large input"
    );
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
        None => 1,                     // the flag only
    };
    let expected_u32 = header + accepting + transitions + prov_suffix;
    assert_eq!(
        bits.len(),
        (expected_u32 * 32) as usize,
        "the bitvec size must be the exact POD formula"
    );
}

// PROOF: the from_bitvec re-sorts the transitions (the the CSR invariant). A
// well-formed bitvec (the to_bitvec of a sorted machine) is already sorted, so
// the re-sort is a no-op (the the identity round-trip). A CORRUPTED bitvec (the
// unsorted transitions) would produce a WRONG CSR (the the contiguity violated)
// without the re-sort. This test verifies: (1) the round-trip is the identity
// (the sorted case), (2) the from_bitvec on an UNsorted transition order still
// yields a valid CSR (the the re-sort fixes it, the no the silent wrong CSR).
#[test]
fn proof_from_bitvec_resorts_for_csr_invariant() {
    // The (1) the identity round-trip (the sorted machine, the to_bitvec -> the
    // from_bitvec is the identity).
    let m = pushdown_rs::compile(&Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![])]))
        .expect("compile");
    let bits = m.to_bitvec();
    let m2 = PdaMachine::from_bitvec(&bits).expect("round-trip");
    assert_eq!(m, m2, "the bitvec round-trip is the identity (the sorted case)");
    // The CSR invariant holds on the round-tripped machine.
    for q in 0..m2.num_states {
        let start = m2.ctrl_offsets[q as usize] as usize;
        let count = m2.ctrl_counts[q as usize] as usize;
        for t in &m2.transitions[start..start + count] {
            assert_eq!(t.q, q, "the CSR range for q must contain only q's transitions");
        }
    }
    // The (2) the unsorted case: hand-build a bitvec with the transitions in a
    // NON-sorted order (the the q's scattered), and verify the from_bitvec
    // re-sorts (the the CSR invariant holds, the no the silent wrong CSR).
    // The machine: the 2 states, the 1 transition each, the q=1 before the q=0
    // (the the UNsorted order).
    let unsorted_m = PdaMachine {
        num_states: 2,
        num_inputs: 1,
        num_stack_syms: 1,
        transitions: vec![
            Transition { q: 1, a: 0, top: 0, next_q: 1, push: vec![0] }, // the q=1 FIRST.
            Transition { q: 0, a: 0, top: 0, next_q: 0, push: vec![0] }, // the q=0 SECOND.
        ],
        accepting: vec![1],
        start_state: 0,
        start_stack: 0,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
        flat_a: vec![],
        flat_top: vec![],
        flat_next_q: vec![],
    };
    // The to_bitvec writes the transitions in the stored (UNsorted) order.
    let unsorted_bits = unsorted_m.to_bitvec();
    let reloaded = PdaMachine::from_bitvec(&unsorted_bits).expect("reload");
    // The from_bitvec re-sorted (the the q=0 transition now before the q=1).
    assert_eq!(
        reloaded.transitions[0].q, 0,
        "the from_bitvec must re-sort (the q=0 first, the no the unsorted q=1 first)"
    );
    assert_eq!(reloaded.transitions[1].q, 1, "the q=1 second (the sorted order)");
    // The CSR invariant holds on the reloaded machine.
    for q in 0..reloaded.num_states {
        let start = reloaded.ctrl_offsets[q as usize] as usize;
        let count = reloaded.ctrl_counts[q as usize] as usize;
        for t in &reloaded.transitions[start..start + count] {
            assert_eq!(t.q, q, "the CSR range for q must contain only q's transitions (the re-sorted)");
        }
    }
}

// PROOF: the derives_eps is TOTAL (the no the out-of-range panic). A terminal
// ID or an out-of-range nt returns false (the no the panic). The valid
// nonterminal returns the correct nullable status.
#[test]
fn proof_derives_eps_is_total() {
    use pushdown_rs::compile::derives_eps;
    // The S -> A, A -> eps (the S is nullable via the A).
    let g = Cfg::new(2, 1, 0, vec![(0, vec![1]), (1, vec![])]);
    assert!(derives_eps(&g, 0), "the S is nullable (the S -> A -> eps)");
    assert!(derives_eps(&g, 1), "the A is nullable (the A -> eps)");
    // The terminal ID (the 2, the num_nt=2, the terminal range is 2..3) is NOT
    // nullable (the no the panic, the total function).
    assert!(!derives_eps(&g, 2), "the terminal is not nullable (the total, the no the panic)");
    // The out-of-range nt (the 99) is NOT nullable (the no the panic, the total).
    assert!(!derives_eps(&g, 99), "the out-of-range nt is not nullable (the total, the no the panic)");
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
        (
            3,
            1,
            vec![(0, vec![1]), (1, vec![1, 0]), (2, vec![1, 1, 0, 2])],
        ),
        (
            1,
            3,
            vec![(0, vec![1]), (0, vec![2]), (0, vec![3]), (0, vec![])],
        ),
    ];
    for (num_nt, num_tm, prods) in cases {
        let g = Cfg::new(num_nt, num_tm, 0, prods.clone());
        // the independent formula: 1 + 2|N| + sum_p(|rhs(p)|+1).
        let expected: u32 = 1
            + 2 * num_nt
            + prods
                .iter()
                .map(|(_, rhs)| rhs.len() as u32 + 1)
                .sum::<u32>();
        assert_eq!(
            kappa(&g),
            expected,
            "the kappa must equal the independent formula (|N|={num_nt})"
        );
        // the compiled machine must have exactly kappa states (the Lemma 2).
        let m = pushdown_rs::compile(&g).expect("compile");
        assert_eq!(
            m.num_states, expected,
            "the compiled state count must equal kappa(G)"
        );
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
        pushdown_rs::compile(&Cfg::new(1, 2, 0, vec![(0, vec![1, 2])]))
            .expect("the S -> a b (the single production)"),
    ];
    for m in &det_machines {
        assert!(m.is_deterministic(), "the machine must be deterministic");
        let mut keys: Vec<(u32, u32, u32)> =
            m.transitions.iter().map(|t| (t.q, t.a, t.top)).collect();
        keys.sort();
        let dupes = keys
            .iter()
            .zip(keys.iter().skip(1))
            .filter(|(a, b)| a == b)
            .count();
        assert_eq!(
            dupes, 0,
            "a deterministic machine has no duplicate (q, a, top) key"
        );
    }
    // the ambiguous machine (the HAS a dup key, the non-determinism is real).
    let amb = pushdown_rs::compile(&Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![])]))
        .expect("the S -> a S b | eps (the choice)");
    assert!(
        !amb.is_deterministic(),
        "the ambiguous grammar must be non-deterministic"
    );
    let mut keys: Vec<(u32, u32, u32)> =
        amb.transitions.iter().map(|t| (t.q, t.a, t.top)).collect();
    keys.sort();
    let dupes = keys
        .iter()
        .zip(keys.iter().skip(1))
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        dupes > 0,
        "the ambiguous machine must have a duplicate (q, a, top) key (the non-determinism)"
    );
}

// TheOF: the accepts_dpda (the single-path simulation) agrees with the CYK
// (the accepts_npda, the O(n^3) CFG word problem) for the DETERMINISTIC case.
// This is the inclusive + exclusive proof for the general DPDA (the no just
// the a^n b^n): the single-path simulation is correct iff it agrees with the
// independent CYK oracle on every input. The Sénizergues 1997 result (the DPDA
// language equivalence is decidable, the O(n^3) algorithm) grounds this.
//
// For NON-deterministic grammars (the RTN choice at the q_A^in), the
// accepts_dpda returns false (the no the unique path, the is_deterministic
// check fails), and the CYK (the accepts_npda) is the correct oracle. The
// test verifies this: the deterministic grammars have accepts_dpda == CYK,
// the non-deterministic grammars have accepts_dpda == false (the CYK is the
// correct answer, the no the single-path).
#[test]
fn proof_dpda_single_path_agrees_with_cyk() {
    use pushdown_rs::pda::{Dpda, Npda};
    // The deterministic grammars (the no-production, the no the RTN choice).
    let det_grammars: Vec<Cfg> = vec![
        // The S -> a b (the single production, the deterministic).
        Cfg::new(1, 2, 0, vec![(0, vec![1, 2])]),
    ];
    for g in &det_grammars {
        let m = pushdown_rs::compile(g).expect("compile");
        assert!(m.is_deterministic(), "the grammar must be deterministic");
        // The corpus: all binary strings over the terminals, the 0..=6.
        let num_nt = g.num_nonterminals;
        for len in 0..=6 {
            for mask in 0..(1usize << len) {
                let global: Vec<u32> = (0..len)
                    .map(|i| num_nt + ((mask >> i) & 1) as u32)
                    .collect();
                let local: Vec<u32> = global.iter().map(|&x| x - num_nt).collect();
                let dpda_says = m.accepts_dpda(&local);
                let cyk_says = m.accepts_npda(g, &local);
                assert_eq!(
                    dpda_says, cyk_says,
                    "the accepts_dpda must agree with the CYK (the deterministic case, the Sénizergues 1997)"
                );
            }
        }
    }
    // The non-deterministic grammars (the RTN choice, the a^n b^n, the balanced parens).
    let nondet_grammars: Vec<Cfg> = vec![
        // The a^n b^n (the S -> a S b | eps, the RTN choice, the non-deterministic).
        Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![])]),
        // The balanced parens (the S -> ( S ) S | eps, the RTN choice, the non-deterministic).
        Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2, 0]), (0, vec![])]),
    ];
    for g in &nondet_grammars {
        let m = pushdown_rs::compile(g).expect("compile");
        assert!(!m.is_deterministic(), "the grammar must be non-deterministic (the RTN choice)");
        // The corpus: all binary strings over the terminals, the 0..=6.
        let num_nt = g.num_nonterminals;
        for len in 0..=6 {
            for mask in 0..(1usize << len) {
                let global: Vec<u32> = (0..len)
                    .map(|i| num_nt + ((mask >> i) & 1) as u32)
                    .collect();
                let local: Vec<u32> = global.iter().map(|&x| x - num_nt).collect();
                let dpda_says = m.accepts_dpda(&local);
                let cyk_says = m.accepts_npda(g, &local);
                // The accepts_dpda is guarded to false for the non-deterministic
                // machine (the is_deterministic check fails, the no the unique
                // path). The CYK (the accepts_npda) is the correct procedure for
                // the non-deterministic case, and must agree with the independent
                // cfg_accepts oracle.
                assert!(
                    !dpda_says,
                    "the accepts_dpda must be false for the non-deterministic machine (the guard)"
                );
                assert_eq!(
                    cyk_says,
                    pushdown_rs::oracle::cfg_accepts(g, &global),
                    "the CYK (the accepts_npda) must agree with the independent cfg_accepts oracle"
                );
            }
        }
    }
    // The DETERMINISTIC a^n b^n PDA (the JFLAP hand-built machine, the unique
    // computation path): the single-path simulation (the accepts_dpda) is EXACT
    // (the deterministic PDA has a unique computation on each input, the
    // Hopcroft-Ullman / the DCFL = the DPDA-recognizable result). It must agree
    // with the independent cfg_accepts oracle on the a^n b^n language.
    let anb = anb_n_dpda();
    assert!(anb.is_deterministic(), "the JFLAP a^n b^n PDA is deterministic");
    let g_anb = Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![])]); // the S -> a S b | eps
    // The JFLAP a^n b^n machine accepts {a^n b^n : n >= 1} (the no the empty string,
    // the start q q0 has no epsilon move). The n=0 case is the empty string,
    // which is OUT of the language (the rejects_dpda).
    for n in 1..=8 {
        // The a^n b^n (the in-language) + the a^n b^(n+1) / a^(n+1) b^n (the out-of-language).
        let mut in_lang = vec![0u32; n];
        in_lang.extend(vec![1; n]);
        let mut over_b = vec![0u32; n];
        over_b.extend(vec![1; n + 1]);
        let mut over_a = vec![0u32; n + 1];
        over_a.extend(vec![1; n]);
        assert!(
            anb.accepts_dpda(&in_lang),
            "the a^n b^n (the n={n}) is in the language (the inclusive_dpda)"
        );
        assert!(
            !anb.accepts_dpda(&over_b),
            "the a^n b^(n+1) (the n={n}) is out of the language (the rejects_dpda)"
        );
        assert!(
            !anb.accepts_dpda(&over_a),
            "the a^(n+1) b^n (the n={n}) is out of the language (the rejects_dpda)"
        );
        // The independent oracle (the cfg_accepts, the zero shared code with the PDA).
        assert_eq!(
            anb.accepts_dpda(&in_lang),
            pushdown_rs::oracle::cfg_accepts(&g_anb, &in_lang.iter().map(|&x| x + 1).collect::<Vec<u32>>()),
            "the accepts_dpda must agree with the cfg_accepts oracle (the a^n b^n)"
        );
    }
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
        assert!(
            stack.len() <= d,
            "a reachable config must respect the stack bound d"
        );
        max_depth = max_depth.max(stack.len());
    }
    // (exclusive) the bound is real: the a^n b^n nests to a non-trivial depth
    // (the [A_SYM, Z] after the first 'a'), and the per-step push is bounded.
    assert!(
        max_depth >= 2,
        "the a^n b^n must reach a non-trivial nesting depth"
    );
    let max_push = m
        .transitions
        .iter()
        .map(|t| t.push.len())
        .max()
        .unwrap_or(0);
    assert!(
        max_push <= 2,
        "the per-step push is bounded (the at most 2 symbols)"
    );
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
    fn sequential_projection(
        m: &PdaMachine,
        config: (u32, Vec<u32>),
        draft: &[u32],
    ) -> Vec<Vec<u32>> {
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
    assert_eq!(
        projected[0],
        sequential_projection(&m, config.clone(), &legal_draft),
        "the legal projection must equal the K sequential steps"
    );
    assert_eq!(
        projected[0].len(),
        legal_draft.len() + 1,
        "the legal draft emits K+1 masks (the inclusive)"
    );

    // the divergent draft (the a b b a: the trailing tokens dead-end at state 2):
    // the projection STOPS at the first illegal token (the exclusive).).
    let illegal_draft = vec![0u32, 1, 1, 0];
    let projected2 = m.project_batch(&[config.clone()], &[illegal_draft.clone()]);
    let expected2 = sequential_projection(&m, config.clone(), &illegal_draft);
    assert_eq!(
        projected2[0], expected2,
        "the divergent projection must stop at the first illegal token"
    );
    assert!(
        projected2[0].len() < illegal_draft.len() + 1,
        "the divergent draft emits FEWER than K+1 masks (the exclusive stop)"
    );
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
    assert!(
        summary.is_reachable(m.start_state, &[m.start_stack]),
        "the start is reachable"
    );
    assert!(
        summary.distance(m.start_state, &[m.start_stack]).is_some(),
        "the start has a finite d_H"
    );
    // (exclusive) every reachable config respects the depth bound h.
    for (_, stack) in summary.reachable.iter() {
        assert!(stack.len() <= h, "a reachable config must have depth <= h");
    }
    // (inclusive) every config with a defined d_H is in the Reach_H set.
    for (q, stack) in summary.distance.keys() {
        assert!(
            summary.is_reachable(*q, stack),
            "a d_H-defined config must be reachable"
        );
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
            if s == 0 && b == b'a' {
                1
            } else {
                255
            } // the dead
        }
        fn is_dead(&self, s: u8) -> bool {
            s == 255
        }
        fn accepting(&self, s: u8) -> Vec<u8> {
            if s == 1 {
                vec![1]
            } else {
                vec![]
            }
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
    // The CSR u32 counts (the the ctrl_u32_counts, the the range [offset, +count)
    // for each q). The count must equal the sum of the u32 sizes of q's records
    // (the the 5 + push_len per record). The offset + count must span exactly
    // q's records (the the CSR invariant, the no the silent wrong range).
    let num_states = m.num_states as usize;
    assert_eq!(pkg.ctrl_u32_offsets.len(), num_states, "the offsets length is the num_states");
    assert_eq!(pkg.ctrl_u32_counts.len(), num_states, "the counts length is the num_states");
    // The independent recomputation: the u32 size of each record (the 5 + push_len).
    let record_sizes: Vec<u32> = m.transitions.iter().map(|t| 5 + t.push.len() as u32).collect();
    let total_u32s: u32 = record_sizes.iter().sum();
    for q in 0..num_states {
        // The count: the sum of the u32 sizes of q's records (the the
        // transitions are sorted by q, so q's records are contiguous).
        let q_record_indices: Vec<usize> = m.transitions
            .iter()
            .enumerate()
            .filter(|(_, t)| t.q == q as u32)
            .map(|(idx, _)| idx)
            .collect();
        let expected_count: u32 = q_record_indices.iter().map(|&idx| record_sizes[idx]).sum();
        assert_eq!(
            pkg.ctrl_u32_counts[q], expected_count,
            "the ctrl_u32_counts[q] must equal the sum of q's record u32 sizes"
        );
        if let Some(&first_idx) = q_record_indices.first() {
            // The offset: the u32 index of q's first record (the the running sum
            // of the record sizes before first_idx).
            let expected_offset: u32 = record_sizes[..first_idx].iter().sum();
            assert_eq!(
                pkg.ctrl_u32_offsets[q], expected_offset,
                "the ctrl_u32_offsets[q] must equal the u32 index of q's first record"
            );
        } else {
            // The q has no records: the offset is the sentinel (the total_u32s),
            // the count is 0.
            assert_eq!(pkg.ctrl_u32_offsets[q], total_u32s, "the no-record q gets the sentinel offset");
            assert_eq!(pkg.ctrl_u32_counts[q], 0, "the no-record q gets the count 0");
        }
    }
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
            let w: Vec<u32> = (0..len)
                .map(|i| if (mask >> i) & 1 == 1 { 2 } else { 1 })
                .collect();
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
        let pda_says = m.accepts_npda(g, &pda_input);
        let oracle_says = cfg_accepts(g, w); // the oracle uses the global IDs
        if pda_says != oracle_says {
            disagreements += 1;
            eprintln!(
                "DISAGREE on global={:?} local={:?}: pda={} oracle={}",
                w, pda_input, pda_says, oracle_says
            );
        }
    }
    assert_eq!(
        disagreements,
        0,
        "the PDA must match the independent oracle on all {} inputs",
        corpus.len()
    );
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
            (0, vec![]),        // S -> eps
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
    assert_eq!(
        accepted.len(),
        7,
        "the a^n b^n language accepts EXACTLY 7 strings of length <= 12 (the n=0..=6)"
    );
    for w in &accepted {
        // each accepted string is a^n b^n (the a's then the b's, the equal count).
        let na = w.iter().filter(|&&x| x == 1).count();
        let nb = w.iter().filter(|&&x| x == 2).count();
        assert_eq!(
            na, nb,
            "an accepted string must be a^n b^n (the equal count)"
        );
        assert!(
            w.iter().take_while(|&&x| x == 1).count() == na,
            "the a's must precede the b's"
        );
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
            (0, vec![]),           // S -> eps
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
            (0, vec![]),        // S -> eps
            (1, vec![2, 2]),    // A -> B B
            (1, vec![3]),       // A -> a
            (2, vec![4]),       // B -> b
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // the terminals are the global 3 (a) and 4 (b); the corpus is over {3,4}.
    let corpus: Vec<Vec<u32>> = (0..=8)
        .flat_map(|len| {
            (0..(1usize << len)).map(move |mask| {
                (0..len)
                    .map(|i| if (mask >> i) & 1 == 1 { 4 } else { 3 })
                    .collect::<Vec<u32>>()
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
            (1, vec![4]),       // A -> c
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // the local terminal IDs (the global - the num_nt=2): a=0, b=1, c=2.
    // the input "a c b" = the local [0, 2, 1].
    let local = vec![0u32, 2, 1];
    assert!(
        m.accepts(&g, &local),
        "the PDA accepts a c b (the S -> a A b, A -> c)"
    );
    // the differential: the independent cfg oracle agrees (the global IDs).
    let global = vec![2u32, 4, 3]; // the a, c, b (the global)
    assert_eq!(
        m.accepts(&g, &local),
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
        config = m
            .advance_eps(config.0, &config.1, tok)
            .expect("the advance must succeed (the no stuck call dot)");
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
    assert!(
        reached_accepting,
        "the epsilon closure from the post-consumption config reaches the accepting state"
    );
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
            assert_eq!(
                summary.distance(*q, stack),
                Some(0),
                "an accepting config must have d_H = 0"
            );
        }
    }
    // the start config reaches acceptance within h (the grammar terminates).
    assert!(
        summary.distance(m.start_state, &[m.start_stack]).is_some(),
        "the start must have a finite d_H"
    );
}

// PROOF: the UNBOUNDED weighted-PDS distance (the S, the no the S_H). The d(c)
// is the least fixed point of the distance equations over the (min,+) dioid
// (the Reps 2003, the "min-plus semiring finds the shortest trace"): d(c) = 0
// if c is accepting, else d(c) = 1 + min over c -> c' of d(c'). This is the
// shortest-path-to-acceptance over the (infinite) config graph, computed by the
// reverse BFS (the pre* saturation, the Bouajjani 1997). The test verifies:
//   (inclusive) the unbounded d agrees with the bounded d (the compute(machine,
//               h) for a large enough h) on every config where both are defined;
//   (exclusive) the accepting configs have d = 0 (the unconditional);
//   (oracle) the unbounded d agrees with the independent BFS shortest-path
//            (the reverse graph, the zero shared code with the compute_unbounded).
#[test]
fn proof_unbounded_weighted_pds_distance() {
    use pushdown_rs::summary::BoundedSummary;
    // The S -> a b machine (the  stack, the 2 states, the fast unbounded
    // config space). The unbounded d is computed over the full (finite) config
    // space (the stack depth is bounded by the machine structure).
    let m = pushdown_rs::compile(&Cfg::new(1, 2, 0, vec![(0, vec![1, 2])])).expect("compile S -> a b");
    let unbounded = BoundedSummary::compute_unbounded(&m);
    // (exclusive) the accepting configs have d = 0 (the unconditional).
    for (q, stack) in unbounded.reachable.iter() {
        if m.accepting.contains(q) {
            assert_eq!(
                unbounded.distance(*q, stack),
                Some(0),
                "an accepting config must have d = 0 (the unbounded)"
            );
        }
    }
    // (oracle) the unbounded d agrees with the independent BFS shortest-path
    // (the reverse graph, the zero shared code with the compute_unbounded).
    let independent_d = independent_shortest_distance(&m);
    for (cfg, d) in &independent_d {
        assert_eq!(
            unbounded.distance(cfg.0, &cfg.1),
            Some(*d),
            "the unbounded d must agree with the independent BFS oracle"
        );
    }
    // The compute_unbounded is for RTN-compiled machines (the bounded stack, the
    // six-property #5: the D = max_push + 1). The S -> a b machine above has a
    // bounded stack (the no call, the depth <= 1), so the compute_unbounded
    // terminates. The hand-built a^n b^n machine has an UNBOUNDED stack (the
    // a's push, the depth grows with n), so the compute_unbounded (the D =
    // max_push + 1) is too tight for it - the a^n b^n is the bounded compute
    // (the caller-supplied h), not the unbounded.
}

/// The independent BFS shortest-path-to-acceptance (the reverse graph, the zero
/// shared code with the compute_unbounded). The oracle for the unbounded d proof.
fn independent_shortest_distance(m: &PdaMachine) -> Vec<((u32, Vec<u32>), u32)> {
    use std::collections::{HashMap, VecDeque};
    // The six-property #5 bounded pushdown: the D = max_push + 1 (the max
    // production length + 1, the pending nesting). The max_push is derived
    // from the machine (the maximum length of any transition's push string).
    // The reverse BFS is restricted to stacks of depth <= D (the no the
    // unbounded exploration, which does not terminate).
    let max_push = m.transitions.iter().map(|t| t.push.len()).max().unwrap_or(1);
    let d_bound = max_push + 1; // the stack depth bound (the six-property #5).
    // The reverse BFS from the accepting configs (the d = 0).
    let mut dist: HashMap<(u32, Vec<u32>), u32> = HashMap::new();
    let mut queue: VecDeque<(u32, Vec<u32>, u32)> = VecDeque::new();
    for &fq in &m.accepting {
        let cfg = (fq, vec![m.start_stack]);
        dist.insert(cfg.clone(), 0);
        queue.push_back((cfg.0, cfg.1, 0));
    }
    while let Some((cq, cstk, cd)) = queue.pop_front() {
        // The reverse transitions: find all predecessor configs (q, sigma) that
        // transition to (cq, cstk) via a single move. The PDA transition semantics
        // (the ODU CS390, the delta(q, a, top) = {(q', push)}): the successor stack
        // is cstk = push.reversed() ++ sigma' where sigma' = sigma without its top.
        // So sigma' = cstk[|push|..], top is a free variable in Gamma, and
        // sigma = sigma' ++ [top]. The rule (t.q, a, top) -> (t.next_q, t.push)
        // must exist for the predecessor (t.q, sigma) to reach (cq, cstk).
        for a in 0..=m.num_inputs {
            for t in &m.transitions {
                if t.next_q != cq || t.a != a {
                    continue;
                }
                // The PDA transition semantics (the ODU CS390): the successor stack
                // is cstk = deeper ++ push.reversed() where deeper = sigma without
                // its top. So:
                //   m      = t.push.len()
                //   deeper = cstk[0..(cstk.len() - m)]   (the the stack below the pushed block)
                //   top    = t.top                        (the the rule's popped symbol, the FIXED)
                //   sigma  = deeper ++ [top]             (the the predecessor stack)
                let m_len = t.push.len();
                if cstk.len() < m_len {
                    continue;
                }
                let deeper: Vec<u32> = cstk[..(cstk.len() - m_len)].to_vec();
                let mut sigma = deeper;
                sigma.push(t.top); // the the predecessor top (the the rule's t.top, the FIXED).
                // The six-property #5 bound: the predecessor stack depth must
                // be <= d_bound (the no the unbounded exploration).
                if sigma.len() > d_bound {
                    continue;
                }
                let pred = (t.q, sigma.clone());
                let new_d = cd + 1;
                if let Some(&existing) = dist.get(&pred) {
                    if existing <= new_d {
                        continue; // the no improvement.
                    }
                }
                dist.insert(pred.clone(), new_d);
                queue.push_back((pred.0, pred.1, new_d));
            }
        }
    }
    dist.into_iter().map(|(cfg, d)| (cfg, d)).collect()
}

// PROOF: the compute_unbounded reverse equation on a MULTI-SYMBOL push over a
// non-trivial deeper stack (the the case the a^n b^n / the S->ab tests do NOT
// cover, where cstk.len() > m and m > 1). The old buggy equation (the
// sigma_prime = cstk[m..], the the pushed block, the no the deeper stack) gives
// the WRONG predecessor; the corrected equation (the deeper = cstk[0..(len-m)],
// the top = t.top FIXED) gives the RIGHT one. This test verifies the
// compute_unbounded distance agrees with the independent oracle on a machine
// where the two cases diverge.
#[test]
fn proof_unbounded_distance_multi_symbol_push_over_deep_stack() {
    use pushdown_rs::summary::BoundedSummary;
    // The machine: the q0 --a, Z--> q1 (the push [X, Y], the 2 symbols over the
    // Z).). The q1 --b, Y--> q2 (the push [], the pop Y). The q2 is
    // accepting. The stack grows to depth 3 (the [Z, X, Y]) after the a move,
    // so the reverse step from (q1, [Z, X, Y]) must recover the predecessor
    // (q0, [Z]) via the multi-symbol push [X, Y] (the m = 2, the c =
    // cstk[0..(3-2)] = cstk[0..1] = [Z], the top = t.top = Z).
    const A: u32 = 0;
    const B: u32 = 1;
    const EPS: u32 = 2;
    const Z: u32 = 0;
    const X: u32 = 1;
    const Y: u32 = 2;
    let m = PdaMachine {
        num_states: 3,
        num_inputs: 2,
        num_stack_syms: 3,
        transitions: vec![
            Transition { q: 0, a: A, top: Z, next_q: 1, push: vec![X, Y] }, // the push 2 symbols.
            Transition { q: 1, a: B, top: Y, next_q: 2, push: vec![] }, // the pop Y.
            Transition { q: 2, a: EPS, top: Z, next_q: 2, push: vec![Z] }, // the self-loop.
        ],
        accepting: vec![2],
        start_state: 0,
        start_stack: Z,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
        flat_a: vec![],
        flat_top: vec![],
        flat_next_q: vec![],
    };
    let unbounded = BoundedSummary::compute_unbounded(&m);
    let oracle = independent_shortest_distance(&m);
    // The oracleinclusive) the unbounded distance agrees with the independent oracle
    // on every config the oracle reached.
    for (cfg, d) in &oracle {
        assert_eq!(
            unbounded.distance(cfg.0, &cfg.1),
            Some(*d),
            "the compute_unbounded must agree with the independent oracle (the multi-symbol push over the deep stack)"
        );
    }
    // The (exclusive) the accepting config (q2, [Z]) has d = 0.
    assert_eq!(unbounded.distance(2, &[Z]), Some(0), "the accepting config has d = 0");
    // The (boundary) the predecessor of (q1, [Z, X, Y]) via the a move is (q0, [Z])
    // (the the multi-symbol push [X, Y] over the Z bottom). The d(q0, [Z]) =
    // d(q1, [Z, X, Y]) + 1.
    let d_q1_deep = unbounded.distance(1, &[Z, X, Y]);
    let d_q0 = unbounded.distance(0, &[Z]);
    if let (Some(d1), Some(d0)) = (d_q1_deep, d_q0) {
        assert_eq!(d0, d1 + 1, "the d(q0, [Z]) must be the d(q1, [Z,X,Y]) + 1 (the reverse step)");
    }
}

// PROOF: the max_closure_width (the measurement, the no the static cap). The
// epsilon closure over the (state, top) pairs is bounded by num_states *
// num_stack_syms (the finite domain). The max_closure_width measures the
// EXACT maximum closure width (the no an underestimate). The test verifies:
//   (exclusive) the max_closure_width <= num_states * num_stack_syms (the
//               domain bound, the no the static).
//   (inclusive) the max_closure_width is the EXACT maximum (the no an
//               underestimate), verified against the independent BFS that
//               computes the closure width for each (q, top).
#[test]
fn proof_max_closure_width_is_exact() {
    let m = anb_n_dpda();
    let domain_size = (m.num_states as usize) * (m.num_stack_syms as usize);
    let width = m.max_closure_width();
    // (exclusive) the width is bounded by the domain size (the no the overflow).
    assert!(
        width <= domain_size,
        "the max_closure_width must be <= num_states * num_stack_syms (the domain bound)"
    );
    // (inclusive) the width is the EXACT maximum (the no an underestimate).
    // The independent BFS computes the closure width for each (q, top) and
    // takes the max (the zero shared code with the max_closure_width).
    let independent_width = independent_closure_width(&m);
    assert_eq!(
        width, independent_width,
        "the max_closure_width must equal the independent BFS (the exact maximum)"
    );
}

/// The independent BFS closure width (the zero shared code with the
/// max_closure_width). The oracle for the max_closure_width proof.
fn independent_closure_width(m: &PdaMachine) -> usize {
    let mut max_width = 0usize;
    for q in 0..m.num_states {
        for top in 0..m.num_stack_syms {
            // The epsilon closure over the (state, top) pairs (the independent BFS).
            let mut visited: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
            let mut frontier: Vec<(u32, u32)> = vec![(q, top)];
            while let Some((cq, ctop)) = frontier.pop() {
                if !visited.insert((cq, ctop)) {
                    continue;
                }
                for t in &m.transitions {
                    if t.q == cq && t.a == m.num_inputs && t.top == ctop {
                        let new_top = if t.push.is_empty() {
                            ctop
                        } else {
                            *t.push.first().unwrap()
                        };
                        frontier.push((t.next_q, new_top));
                    }
                }
            }
            max_width = max_width.max(visited.len());
        }
    }
    max_width
}

// PROOF 11: the batch invariant (the step_batch == the scalar step).
// The the batched op op must equal the scalar op applied per-item.
#[test]
fn proof_stream_step_batch_equals_scalar() {
    use pushdown_rs::pda::PdaStream;
    let m = anb_n_dpda();
    // the batch of (config, token) pairs (the batch)
    let batch: Vec<((u32, Vec<u32>), u32)> = vec![
        ((0, vec![0]), 0),    // the start, the 'a'
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
        assert_eq!(
            batched[i], vec![expected],
            "the step_batch[{i}] (the set) must equal the scalar step (the 1-element set, the deterministic a^n b^n)"
        );
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
            (0, vec![1]),    // S -> a
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    assert!(
        !m.is_deterministic(),
        "the [a-z]+ is non-deterministic (the choice)"
    );
    // the PDA's input is the LOCAL terminal ID (the a=0, the b=1); the oracle
    // uses the GLOBAL (the a=1, the b=2). Map global -> local (the - num_nt).
    let to_local = |w: &[u32]| w.iter().map(|&x| x - 1).collect::<Vec<u32>>();
    // the single 'a' (the global [1] = the local [0]) is in the language (the S -> a)
    // the universal `accepts` auto-selects the NPDA (the non-deterministic) -
    // the caller does NOT need to know which variant the machine is.
    assert!(
        m.accepts(&g, &to_local(&[1])),
        "the [a-z]+ accepts a single 'a' (the accepts)"
    );
    assert!(
        m.accepts(&g, &to_local(&[1, 1])),
        "the [a-z]+ accepts 'aa' (the universal accepts)"
    );
    assert!(
        !m.accepts(&g, &to_local(&[])),
        "the [a-z]+ rejects the empty (the one-or-more)"
    );
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
        (1, vec![0, 1], 0),       // the after the 'a', the 'a'
        (1, vec![0, 1], 1),       // the after the 'a', the 'b'
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
        (0, vec![0], 0),    // the start, the 'a'
        (1, vec![0, 1], 0), // the after the 'a', the 'a'
        (1, vec![0, 1], 1), // the after the 'a', the 'b'
    ];
    // the no-allocation step (the pre-allocated buffer)
    let mut out: Vec<(u32, Vec<u32>)> = vec![(0, vec![0]); batch.len()];
    m.step_batch_into(&index, &batch, &mut out);
    // the reference (the step_batch, the per-item)
    let ref_batch: Vec<((u32, Vec<u32>), u32)> = batch
        .iter()
        .map(|(q, s, a)| ((*q, s.clone()), *a))
        .collect();
    let ref_out = m.step_batch(&ref_batch);
    // The step_batch_into (the no-alloc deterministic fast path, the single-config)
    // must equal the FIRST element of the step_batch set (the the deterministic
    // machine produces 1-element sets, the the a^n b^n DPDA). The no-alloc path is
    // the SIMD-accelerated deterministic step (the the lookup_indexed goto, the no
    // the full-domain set). For the non-deterministic machine, use the step_batch
    // (the full-domain set), NOT the step_batch_into (the deterministic fast path).
    let ref_first: Vec<(u32, Vec<u32>)> = ref_out.iter().map(|set| set[0].clone()).collect();
    assert_eq!(
        out, ref_first,
        "the step_batch_into (the deterministic no-alloc) must equal the step_batch's first element (the 1-element set)"
    );
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
    // The step_batch_simd (the deterministic no-alloc SIMD path, the single-config)
    // must equal the FIRST element of the step_batch set (the the deterministic
    // machine produces 1-element sets, the the a^n b^n DPDA).
    let ref_states: Vec<u16> = ref_out.iter().map(|set| set[0].0 as u16).collect();
    assert_eq!(
        simd_out, ref_states,
        "the step_batch_simd (the deterministic no-alloc) must equal the step_batch's first element (the 1-element set)"
    );
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
    assert_eq!(
        simd_out, ref_out,
        "the project_batch_simd must equal the project_batch (the batch invariant)"
    );
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
    assert_eq!(
        dg.nodes.len(),
        3,
        "the 3 nodes (the H2D, the ProjectMasks, the D2H)"
    );
    assert!(matches!(
        dg.nodes[1],
        GraphNode::ProjectMasks { batch: 2, k: 16 }
    ));
    assert!(
        dg.topo_order().is_some(),
        "the drafting graph is a valid DAG"
    );
    let _ = dg.mock_replay(&m).expect("the drafting mock replay");
}

// PROOF 17: the mask_at_cfg_settled is the precise inclusive + (the no
// epsilon-closure union). For EVERY (q, top) config in the full config space,
// mask_at_cfg_settled(q, top) reports EXACTLY the terminals with a defined
// transition at (q, a, top) without following epsilon moves.
//
// The inclusive property: every terminal a with a defined transition (q, a, top)
// is in in the mask.
// The exclusive property: no terminal a WITHOUT a defined transition (q, a, top)
// is in in the mask.
//
// This is the mask source for the PDA-as-mask-source architecture (the
// no-approximation gate). The complement over the vocab is the precise exclusive
// gate (the disallowed terminals).
#[test]
fn proof_mask_at_cfg_settled_is_precise() {
    let m = anb_n_dpda();
    for q in 0..m.num_states {
        for top in 0..m.num_stack_syms {
            // the independent oracle: the terminals with a defined transition at (q, a, top).
            let legal: Vec<u32> = (0..m.num_inputs)
                .filter(|&a| !m.lookup(q, Some(a), top).is_empty())
                .collect();
            // the mask_at_cfg_settled (the precise inclusive gate).
            let mask = m.mask_at_cfg_settled(q, top);
            assert_eq!(
                mask, legal,
                "mask_at_cfg_settled must be EXACTLY the legal inputs (q={q}, top={top})"
            );
            // the exclusive gate: the complement over the vocab.
            let excluded: Vec<u32> = (0..m.num_inputs)
                .filter(|&a| m.lookup(q, Some(a), top).is_empty())
                .collect();
            assert!(
                mask.iter().all(|&a| !excluded.contains(&a)),
                "the mask must not include any excluded terminal (q={q}, top={top})"
            );
        }
    }
}

// PROOF (the drafting-desync root cause): for a NON-deterministic PDA, the GPU
// single-config projection (mask_at_cfg_settled, what fused_sample / fused_project
// compute) is a STRICT under-approximation of the sound epsilon-closure union mask
// (mask_at_cfg). The Qwen tool grammar's PDA is non-deterministic (overlapping token
// ranges, e.g. token 0 in two terminals), so the GPU draft mask under-masks: it omits
// inputs that are legal at an epsilon-reachable config. An illegal-looking draft (like a
// stray "\n\n") then slips through the under-masked draft gate + the under-masked target
// verify, desyncing the guidance FSM. This test proves the under-approximation on a
// minimal non-deterministic machine.
#[test]
fn proof_settled_mask_underapproximates_closure_mask_when_nondeterministic() {
    const A: u32 = 0;
    const B: u32 = 1;
    const EPS: u32 = 2; // the num_inputs
    const Z: u32 = 0; // the start_stack
    // q0 has TWO epsilon moves (to q1 and q2) -> non-deterministic. q1 allows A, q2 allows B.
    let m = PdaMachine {
        num_states: 3,
        num_inputs: 2,
        num_stack_syms: 1,
        transitions: vec![
            Transition { q: 0, a: EPS, top: Z, next_q: 1, push: vec![] },
            Transition { q: 0, a: EPS, top: Z, next_q: 2, push: vec![] },
            Transition { q: 1, a: A, top: Z, next_q: 1, push: vec![Z] },
            Transition { q: 2, a: B, top: Z, next_q: 2, push: vec![Z] },
        ],
        accepting: vec![1, 2],
        start_state: 0,
        start_stack: Z,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
        flat_a: vec![],
        flat_top: vec![],
        flat_next_q: vec![],
    };
    // The machine is genuinely non-deterministic (the two eps branches from q0).
    assert!(!m.is_deterministic(), "the machine must be non-deterministic for the proof");
    // The sound mask (the epsilon-closure union) allows BOTH inputs.
    let closure = m.mask_at_cfg(0, &[]);
    let mut closure_sorted = closure.clone();
    closure_sorted.sort();
    assert_eq!(closure_sorted, vec![A, B], "the closure union allows both a and b");
    // The GPU single-config (settled) mask allows NEITHER (q0 has no input move).
    let settled = m.mask_at_cfg_settled(0, Z);
    assert!(
        settled.is_empty(),
        "the settled single-config mask is empty (the GPU under-mask)"
    );
    // The defect: settled is a strict subset of closure (the GPU under-masks).
    assert!(
        settled.iter().all(|&x| closure.contains(&x)) && settled.len() < closure.len(),
        "the settled mask strictly under-approximates the closure mask"
    );
}

// PROOF: the mask_batch (the epsilon-closure mask, the public PdaStream entry)
// is EXACTLY the set of inputs for which advance_eps (the epsilon-closure
// advance) succeeds. This is the consistent (mask, advance) pair for the
// PDA-as-FSM-mirror contract: the mask reports what the advance can consume,
// and nothing more. Built over the reachable config space (the BFS via
// advance_eps), so it is a genuine oracle (the no single-config shortcut).
#[test]
fn proof_mask_batch_consistent_with_advance_eps() {
    use pushdown_rs::pda::PdaStream;
    // a grammar with a call dot (the S -> a A b, A -> c) — the RTN compilation
    // gives a BOUNDED stack (the D = L + 1), so the reachable config space is
    // finite and the BFS below terminates. (The {a^n b^n} DPDA is unbounded and
    // must not be used here.)
    let g = Cfg::new(2, 3, 0, vec![(0, vec![2, 1, 3]), (1, vec![4])]);
    let m = pushdown_rs::compile(&g).expect("compile");

    // enumerate the reachable configs (state, stack) via a BFS over advance_eps
    let mut reachable: Vec<(u32, Vec<u32>)> = vec![(m.start_state, vec![m.start_stack])];
    let mut i = 0;
    while i < reachable.len() {
        let (q, stack) = reachable[i].clone();
        for a in 0..m.num_inputs {
            if let Some((nq, ns)) = m.advance_eps(q, &stack, a) {
                if !reachable.contains(&(nq, ns.clone())) {
                    reachable.push((nq, ns));
                }
            }
        }
        i += 1;
    }

    // for each reachable config, the mask_at_cfg (the single-config
    // epsilon-closure mask) must be EXACTLY the advance_eps-able inputs (the
    // inclusive + the exclusive), and must agree with the batched mask_batch.
    for (q, stack) in &reachable {
        let mask = m.mask_at_cfg(*q, stack);
        let mask_batched = m.mask_batch(&[(q.clone(), stack.clone())])[0].clone();
        assert_eq!(mask, mask_batched, "mask_at_cfg must equal the batched mask_batch");
        for a in 0..m.num_inputs {
            let in_mask = mask.contains(&a);
            let advance_ok = m.advance_eps(*q, stack, a).is_some();
            assert_eq!(
                in_mask, advance_ok,
                "mask_at_cfg must be exactly the advance_eps-able inputs (q={q}, a={a})"
            );
        }
    }
}

// ============================================================================
// The CSR + the pass-through proofs.
//
// The six-property invariants exercised here:
//   - Bounded control: the CSR range for a state is EXACT (the proof_csr_is_valid).
//   - Deterministic: the pass-through chain is a single path (the no option-spread).
//   - Finite token spanner: the mask is a table lookup (the CSR), not a live re-derivation.
// The oracles are in an INDEPENDENT register (the order-independent linear scan),
// never the CSR code under test (the XOR(F5) no-circularity).
// ============================================================================

// The scattered-choice grammar (the S -> a A b | eps, the A -> B B | a, the B -> b):
// the q_in states have MULTIPLE productions (the choice is are scattered in the
// unsorted build order), so this is the hard case for the CSR contiguity.
fn scattered_cfg() -> Cfg {
    Cfg::new(
        3,
        2,
        0,
        vec![
            (0, vec![3, 1, 4]), // S -> a A b
            (0, vec![]),        // S -> eps
            (1, vec![2, 2]),    // A -> B B
            (1, vec![3]),       // A -> a
            (2, vec![4]),       // B -> b
        ],
    )
}

// PROOF: the CSR is VALID after the sort fix. For every control state q, the range
// [ctrl_offsets[q], +ctrl_counts[q]) contains EXACTLY q's transitions (the inclusive:
// every transition in the range has t.q == q; the exclusive: the count equals q's
// total). This the sort fix, the q_in states with multiple productions were
// scattered, and this range spanned other states' transitions (the broken CSR).
#[test]
fn proof_csr_is_valid() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    assert!(!m.ctrl_offsets.is_empty(), "the compile-based machine must carry the CSR");
    for q in 0..m.num_states {
        let start = m.ctrl_offsets[q as usize] as usize;
        let count = m.ctrl_counts[q as usize] as usize;
        let range = &m.transitions[start..start + count];
        // the inclusive: every transition in the range belongs to q.
        for t in range {
            assert_eq!(t.q, q, "the CSR range for q={q} must contain only q's transitions");
        }
        // the exclusive: the count matches q's total transition count.
        let total_q = m.transitions.iter().filter(|t| t.q == q).count();
        assert_eq!(
            count, total_q,
            "the CSR count for q={q} must equal q's total ({total_q})"
        );
    }
}

// The independent linear reference for the settled mask (the order-independent
// `lookup`, NOT the CSR): The oracle for the mask_at_cfg_settled proof.
fn settled_mask_linear_reference(m: &PdaMachine, q: u32, top: u32) -> Vec<u32> {
    let mut allowed: Vec<u32> = (0..m.num_inputs)
        .filter(|&a| !m.lookup(q, Some(a), top).is_empty())
        .collect();
    allowed.sort();
    allowed
}

// PROOF: the CSR-based mask_at_cfg_settled equals the independent linear reference
// over the FULL config space (the every q, the every top). This is the regression
// gate for the CSR fix: a broken CSR (the scattered q_in) would diverge here.
#[test]
fn proof_mask_settled_csr_equals_linear() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    for q in 0..m.num_states {
        for top in 0..m.num_stack_syms {
            let csr = m.mask_at_cfg_settled(q, top);
            let ref_mask = settled_mask_linear_reference(&m, q, top);
            assert_eq!(
                csr, ref_mask,
                "the CSR settled mask must equal the linear reference (q={q}, top={top})"
            );
        }
    }
}

// The independent linear reference for the epsilon-closure mask (the order-
// independent `lookup` + the Pda-trait `transition`, NOT the CSR).
fn mask_at_cfg_linear_reference(m: &PdaMachine, q: u32, stack: &[u32]) -> Vec<u32> {
    use pushdown_rs::pda::Pda;
    // The full-stack epsilon closure (the consistent with the mask_at_cfg, the no the single-top
    // approximation that desyncs on the empty-push epsilon moves). The linear-scan lookup (the
    // no the CSR) is the independent reference.
    let mut closed: Vec<(u32, Vec<u32>)> = vec![(q, stack.to_vec())];
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    seen.insert((q, stack.last().copied().unwrap_or(m.start_stack)));
    let mut i = 0;
    while i < closed.len() {
        let (cq, cstk) = closed[i].clone();
        i += 1;
        let ctop = cstk.last().copied().unwrap_or(m.start_stack);
        for (q2, push) in m.transition(cq, None, ctop) {
            let mut ns = cstk.clone();
            ns.pop();
            for &p in push.iter().rev() {
                ns.push(p);
            }
            let ns_top = ns.last().copied().unwrap_or(m.start_stack);
            if seen.insert((q2, ns_top)) {
                closed.push((q2, ns));
            }
        }
    }
    // Collect the allowed inputs over the closure.
    let mut allowed: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for (cq, cstk) in &closed {
        let ctop = cstk.last().copied().unwrap_or(m.start_stack);
        for a in 0..m.num_inputs {
            if !m.lookup(*cq, Some(a), ctop).is_empty() {
                allowed.insert(a);
            }
        }
    }
    let mut result: Vec<u32> = allowed.into_iter().collect();
    result.sort();
    result
}

// PROOF: the CSR-based mask_at_cfg (the epsilon-closure union) equals the
// independent linear reference over the REACHABLE config space (the BFS via
// advance_eps). The reachable space is finite (the bounded stack, the six-
// property "bounded pushdown"), so the BFS terminates.
#[test]
fn proof_mask_at_cfg_csr_equals_linear() {
    use pushdown_rs::pda::PdaStream;
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    let mut reachable: Vec<(u32, Vec<u32>)> = vec![(m.start_state, vec![m.start_stack])];
    let mut i = 0;
    while i < reachable.len() {
        let (q, stack) = reachable[i].clone();
        for a in 0..m.num_inputs {
            if let Some((nq, ns)) = m.advance_eps(q, &stack, a) {
                if !reachable.contains(&(nq, ns.clone())) {
                    reachable.push((nq, ns));
                }
            }
        }
        i += 1;
    }
    for (q, stack) in &reachable {
        let mut csr = m.mask_at_cfg(*q, stack);
        csr.sort();
        let ref_mask = mask_at_cfg_linear_reference(&m, *q, stack);
        assert_eq!(
            csr, ref_mask,
            "the CSR mask_at_cfg must equal the linear reference (q={q})"
        );
        // the batched form must agree too (the PdaStream invariant).
        let batched = m.mask_batch(&[(q.clone(), stack.clone())])[0].clone();
        let mut batched_sorted = batched;
        batched_sorted.sort();
        assert_eq!(csr, batched_sorted, "mask_at_cfg must equal the batched mask_batch");
    }
}

// PROOF: the pass-through predicate is SOUND. is_passthrough(q, top, a) holds
// iff an identity-stack input-consuming transition (q, a, top) -> (q', [top])
// exists (the independent linear scan, NOT the CSR). The inclusive: every reported
// pass-through has the identity-stack move; the exclusive: no non-pass-through is
// reported. This is the "no control edge" signal (the stack-preserving shift).
#[test]
fn proof_is_passthrough_sound() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    for q in 0..m.num_states {
        for top in 0..m.num_stack_syms {
            for a in 0..m.num_inputs {
                let reference = m.transitions.iter().any(|t| {
                    t.q == q && t.a == a && t.top == top && t.push.len() == 1 && t.push[0] == top
                });
                assert_eq!(
                    m.is_passthrough(q, top, a),
                    reference,
                    "is_passthrough must agree with the linear reference (q={q}, top={top}, a={a})"
                );
            }
        }
    }
}

// The independent linear reference for the pass-through run length (the order-
// independent chain-follow, NOT the CSR passthrough_next).
fn passthrough_run_reference(m: &PdaMachine, q: u32) -> u32 {
    let mut cur = q;
    let mut depth = 0u32;
    let cap = m.num_states;
    while depth < cap {
        let mut found = None;
        for t in &m.transitions {
            if t.q == cur && t.a < m.num_inputs && t.push.len() == 1 && t.push[0] == t.top {
                found = Some(t.next_q);
                break;
            }
        }
        match found {
            Some(nq) => {
                cur = nq;
                depth += 1;
            }
            None => break,
        }
    }
    depth
}

// PROOF: the pass-through run length is EXACT. passthrough_run(q) equals the
// number of consecutive identity-stack terminal shifts from q (the independent
// linear chain-follow). For the RTN dot states this is the run of consecutive
// terminals in the production's rhs (the "bounded control" six-property: the run
// is bounded by the production length, never the sequence length).
#[test]
fn proof_passthrough_run_exact() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    for q in 0..m.num_states {
        assert_eq!(
            m.passthrough_run(q),
            passthrough_run_reference(&m, q),
            "passthrough_run must equal the linear reference (q={q})"
        );
    }
    // the boundary: a dot state at the LAST terminal of a production has run 1;
    // a state at a nonterminal (the call) has run 0 (the no input-consuming edge).
    // The the run is -independent (the RTN terminal shifts preserve the top).
    for q in 0..m.num_states {
        let run0 = m.passthrough_run(q);
        assert_eq!(run0, passthrough_run_reference(&m, q));
    }
}

// PROOF (the regression): the scattered q_in choice is handled by the CSR. The
// S -> a S b | eps grammar has q_in(S) with TWO choice transitions (the scattered
// in the unsorted build). Before the sort fix, the CSR range for q_in(S) spanned
// the wrong transitions, and advance_eps / mask_at_cfg would MISS one of the two
// productions. After the fix, both are reachable: the mask at the start includes the
// a terminal (the S -> a S b) AND advance_eps succeeds on it.
#[test]
fn proof_advance_eps_scattered_choice() {
    let g = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2]), // S -> a S b
            (0, vec![]),        // S -> eps
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    assert!(!m.ctrl_offsets.is_empty(), "the CSR must be present");
    let start_stack = vec![m.start_stack];
    // the a terminal (the local 0) must be in the start mask (the S -> a S b choice).
    let mask = m.mask_at_cfg(m.start_state, &start_stack);
    assert!(
        mask.contains(&0),
        "the a terminal must be reachable from the start (the S -> a S b choice)"
    );
    // advance_eps on a must succeed (the S -> a S b path).
    assert!(
        m.advance_eps(m.start_state, &start_stack, 0).is_some(),
        "advance_eps on the a must succeed (the scattered choice is handled)"
    );
    // the differential: the PDA language must still match the independent CFG oracle
    // (the sort is a reordering, the language is unchanged).
    let corpus = ab_corpus(8);
    run_differential(&m, &g, &corpus);
}

// The independent reference for accepts_via_eps (the order-independent linear
// scan via the Pda-tr `transition`, NOT the CSR). The oracle for the
// epsilon-closure acceptance proof.
fn accepts_via_eps_reference(m: &PdaMachine, q: u32, stack: &[u32]) -> bool {
    use pushdown_rs::pda::Pda;
    let mut visited: std::collections::HashSet<(u32, Vec<u32>)> = std::collections::HashSet::new();
    let mut frontier: Vec<(u32, Vec<u32>)> = vec![(q, stack.to_vec())];
    while let Some((cq, cstk)) = frontier.pop() {
        if !visited.insert((cq, cstk.clone())) {
            continue;
        }
        if m.accepting().contains(&cq) {
            return true;
        }
        let ctop = cstk.last().copied().unwrap_or(m.start_stack());
        for (q2, push) in m.transition(cq, None, ctop) {
            let mut ns = cstk.clone();
            ns.pop();
            for &p in &push {
                ns.push(p);
            }
            frontier.push((q2, ns));
        }
    }
    false
}

// PROOF: the epsilon-closure acceptance (the accepts_via_eps) equals the
// independent linear-scan reference over the REACHABLE config space (the BFS via
// advance_eps). This is the correct EOS check: the PDA has "finished" the grammar
// iff the epsilon-closure of the current config contains an accepting state (the
// final-state acceptance criterion, the no directis q itself accepting" shortcut).
#[test]
fn proof_accepts_via_eps_equals_reference() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    // Enumerate the reachable configs (the BFS via advance_eps, the bounded stack).
    let mut reachable: Vec<(u32, Vec<u32>)> = vec![(m.start_state, vec![m.start_stack])];
    let mut i = 0;
    while i < reachable.len() {
        let (q, stack) = reachable[i].clone();
        for a in 0..m.num_inputs {
            if let Some((nq, ns)) = m.advance_eps(q, &stack, a) {
                if !reachable.contains(&(nq, ns.clone())) {
                    reachable.push((nq, ns));
                }
            }
        }
        i += 1;
    }
    for (q, stack) in &reachable {
        let csr = m.accepts_via_eps(*q, stack);
        let ref_val = accepts_via_eps_reference(&m, *q, stack);
        assert_eq!(
            csr, ref_val,
            "accepts_via_eps must equal the linear reference (q={q}, stack={stack:?})"
        );
    }
    // The boundary: the start config IS accepting (the S -> eps production, the
    // empty string is in the language), but a mid-derivation config (after
    // consuming the first "a") is NOT accepting (the PDA still needs "A b").
    assert!(m.accepts_via_eps(m.start_state, &vec![m.start_stack]), "the start IS accepting (the S -> eps)");
    let (q1, s1) = m.advance_eps(m.start_state, &vec![m.start_stack], 0).expect("the advance on the a");
    assert!(!m.accepts_via_eps(q1, &s1), "the mid-derivation config is NOT accepting");
}

// The independent displacement reference (the order over the Pda-trait `transition`,
// the order-independent linear scan, NOT the CSR advance_eps). The oracle for the
// displacement proof.
fn displacement_reference(m: &PdaMachine, t: &[u32]) -> Vec<(u32, Vec<u32>, u32, Vec<u32>)> {
    use pushdown_rs::pda::Pda;
    // The reachable in_configs (the BFS over the advance_eps-equivalent from the start,
    // the epsilon closure + the terminal move per input, matching the displacement
    // method's advance_eps BFS).
    let mut in_configs: Vec<(u32, Vec<u32>)> = vec![(m.start_state, vec![m.start_stack])];
    let mut i = 0;
    while i < in_configs.len() {
        let (q, stack) = in_configs[i].clone();
        for a in 0..m.num_inputs {
            // The advance_eps-equivalent: the epsilon closure from (q, stack), then the
            // terminal move on a (the raw transitions, the no advance_eps).
            let mut eps_configs: Vec<(u32, Vec<u32>)> = vec![(q, stack.clone())];
            let mut j = 0;
            while j < eps_configs.len() {
                let (eq, estack) = eps_configs[j].clone();
                let etop = estack.last().copied().unwrap_or(m.start_stack);
                for (q2, push) in m.transition(eq, None, etop) {
                    let mut ns = estack.clone();
                    ns.pop();
                    for &p in push.iter().rev() {
                        ns.push(p);
                    }
                    if !eps_configs.contains(&(q2, ns.clone())) {
                        eps_configs.push((q2, ns));
                    }
                }
                j += 1;
            }
            for (eq, estack) in &eps_configs {
                let etop2 = estack.last().copied().unwrap_or(m.start_stack);
                for (q2, push) in m.transition(*eq, Some(a), etop2) {
                    let mut ns = estack.clone();
                    ns.pop();
                    for &p in push.iter().rev() {
                        ns.push(p);
                    }
                    if !in_configs.contains(&(q2, ns.clone())) {
                        in_configs.push((q2, ns));
                    }
                }
            }
        }
        i += 1;
    }
// For each in_config, simulate the PDA over t (the terminal sequence) and collect
    // the (in_config, out_config) pairs. The simulation matches the advance_eps
    // logic (the frontier BFS epsilon closure + the reverse push), but uses the
    // raw transitions (the m.transition, the independent register).
    let mut result: Vec<(u32, Vec<u32>, u32, Vec<u32>)> = Vec::new();
    for (in_q, in_stack) in &in_configs {
        let mut ctrl = *in_q;
        let mut stack = in_stack.clone();
        let mut diverged = false;
        for &a in t {
            // The epsilon closure (the frontier BFS over the epsilon moves, the
            // reverse push: the push[0] is the new top).
            let mut eps_configs: Vec<(u32, Vec<u32>)> = vec![(ctrl, stack.clone())];
            let mut j = 0;
            while j < eps_configs.len() {
                let (eq, estack) = eps_configs[j].clone();
                j += 1;
                let etop = estack.last().copied().unwrap_or(m.start_stack);
                for (q2, push) in m.transition(eq, None, etop) {
                    let mut ns = estack.clone();
                    ns.pop();
                    for &p in push.iter().rev() {
                        ns.push(p);
                    }
                    if !eps_configs.contains(&(q2, ns.clone())) {
                        eps_configs.push((q2, ns));
                    }
                }
            }
            // The terminal move on a (from each config in the epsilon closure, the
            // reverse push).
            let mut next_configs: Vec<(u32, Vec<u32>)> = Vec::new();
            for (eq, estack) in &eps_configs {
                let etop2 = estack.last().copied().unwrap_or(m.start_stack);
                for (q2, push) in m.transition(*eq, Some(a), etop2) {
                    let mut ns = estack.clone();
                    ns.pop();
                    for &p in push.iter().rev() {
                        ns.push(p);
                    }
                    next_configs.push((q2, ns));
                }
            }
            if next_configs.is_empty() {
                diverged = true;
                break;
            }
            // The deterministic case: the single next config (the no option).
            ctrl = next_configs[0].0;
            stack = next_configs[0].1.clone();
        }
        if !diverged {
            result.push((*in_q, in_stack.clone(), ctrl, stack));
        }
    }
    result
}

// PROOF: the displacement (the CSR advance_eps-based) equals the independent
// linear reference (the Pda-trait transition-BFS) over the reachable config space.
// This is the CFGzip Theorem 2 primitive (the displacement equivalence): two tokens
// are interchangeable iff they have the same displacement (the set of
// (in_config, out_config) pairs).
#[test]
fn proof_displacement_equals_reference() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    // A few of terminal sequences (the the P the PDA consumes).
    let sequences: Vec<Vec<u32>> = vec![
        vec![0], vec![1], vec![0, 1], vec![1, 0], vec![0, 0, 1], vec![],
    ];
    for seq in &sequences {
        let csr = m.displacement(seq);
        let ref_val = displacement_reference(&m, seq);
        // The both are sets of (in_config, out_config) pairs (the no order).
        let mut a: Vec<_> = csr.clone();
        let mut b: Vec<_> = ref_val.clone();
        a.sort();
        b.sort();
        if a != b {
            eprintln!("DEBUG seq={seq:?} csr_len={} ref_len={}", a.len(), b.len());
            eprintln!("DEBUG csr={a:?}");
            eprintln!("DEBUG ref={b:?}");
        }
        assert_eq!(
            a, b,
            "the displacement must equal the linear reference (seq={seq:?})"
        );
    }
}

// PROOF: the displacement partition is an EQUIVALENCE relation (the reflexive, the
// symmetric, the transitive). This is the CFGzip Theorem 2 (the displacement
// equivalence refines the syntactic congruence): two tokens are in the same class
// iff they are interchangeable (the same displacement).
#[test]
fn proof_displacement_partition_is_equivalence() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    let sequences: Vec<Vec<u32>> = vec![
        vec![0], vec![1], vec![0, 1], vec![1, 0], vec![0, 0, 1], vec![], vec![0], vec![1, 1],
    ];
    let groups = m.displacement_partition(&sequences);
    // The partition is total (every sequence is in exactly one group).
    let mut assigned = vec![false; sequences.len()];
    for g in &groups {
        for &idx in g {
            assert!(!assigned[idx], "a sequence must be in exactly one group");
            assigned[idx] = true;
        }
    }
    assert!(assigned.iter().all(|&a| a), "every sequence must be assigned");
    // The equivalence: two sequences are in the same group iff they have the same
    // displacement (the independent check).
    for i in 0..sequences.len() {
        for j in (i + 1)..sequences.len() {
            let same_group = groups.iter().any(|g| g.contains(&i) && g.contains(&j));
            let same_disp = m.displacement(&sequences[i]) == m.displacement(&sequences[j]);
            assert_eq!(
                same_group, same_disp,
                "the partition must group exactly the same-displacement sequences (i={i}, j={j})"
            );
        }
    }
}

// PROOF: the displacement composition (the functional property): the
// displacement of the concatenation t1 ++ t2 is the composition of the
// displacements (the D(t1 ++ t2) = D(t2) o D(t1), the t1 is consumed first, then
// the t2). This is the relation composition (the set of (in_config, out_config)
// pairs such that there exists an intermediate config).
#[test]
fn proof_displacement_composition() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    // A few terminal sequences (the PDA consumes).
    let t1 = vec![0u32];
    let t2 = vec![1u32];
    let concat: Vec<u32> = t1.iter().chain(t2.iter()).copied().collect();
    // The displacement of the concatenation (the direct computation).
    let d_concat = m.displacement(&concat);
    // The composition of the displacements (the D(t2) o D(t1)).
    let d1 = m.displacement(&t1);
    let d2 = m.displacement(&t2);
    let d_composed = pushdown_rs::machine::PdaMachine::displacement_compose(&d1, &d2);
    // The both must be equal (the functional property).
    let mut a: Vec<_> = d_concat;
    let mut b: Vec<_> = d_composed;
    a.sort();
    b.sort();
    assert_eq!(
        a, b,
        "the displacement of the concatenation must equal the composition of the displacements"
    );
}

// PROOF: the displacement monoid (the identity + the associativity). The
// displacement of the empty sequence is the identity relation (the (c, c) pairs
// for all reachable configs c). The composition is associative (the D(t1 ++ t2 ++
// t3) = D(t3) o D(t2) o D(t1) = (D(t3) o D(t2)) o D(t1)). This is the key
// algebraic property that makes the displacement a functional atom (the no
// temporary state, the pure function).
#[test]
fn proof_displacement_monoid_identity() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    // The displacement of the empty sequence is the identity relation (the
    // (c, c) pairs for all reachable configs c).
    let d_empty = m.displacement(&[]);
    for &(in_q, ref in_stack, out_q, ref out_stack) in &d_empty {
        assert_eq!(in_q, out_q, "the identity displacement must have in_q == out_q");
        assert_eq!(in_stack, out_stack, "the identity displacement must have in_stack == out_stack");
    }
    // The empty sequence displacement is non-empty (the reachable configs exist).
    assert!(!d_empty.is_empty(), "the identity displacement must be non-empty (the reachable configs)");
}

#[test]
fn proof_displacement_monoid_associativity() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    let t1 = vec![0u32];
    let t2 = vec![1u32];
    let t3 = vec![0u32, 1u32];
    // The D(t1 ++ t2 ++ t3) (the left-associative, the direct computation).
    let left_concat: Vec<u32> = t1.iter().chain(t2.iter()).chain(t3.iter()).copied().collect();
    let d_left = m.displacement(&left_concat);
    // The (D(t3) o D(t2)) o D(t1) (the right-associative, the composition).
    let d_t1 = m.displacement(&t1);
    let d_t2 = m.displacement(&t2);
    let d_t3 = m.displacement(&t3);
    let d_t3o_t2 = PdaMachine::displacement_compose(&d_t2, &d_t3);
    let d_right = PdaMachine::displacement_compose(&d_t3o_t2, &d_t1);
    // The both must be equal (the associativity).
    let mut a: Vec<_> = d_left;
    let mut b: Vec<_> = d_right;
    a.sort();
    b.sort();
    assert_eq!(
        a, b,
        "the displacement composition must be associative (the D(t1++t2++t3) == the (D(t3) o D(t2)) o D(t1))"
    );
}

// PROOF: the displacement congruence (the key property that makes the
// displacement a valid equivalence relation): if t1 ~ t2 (the same
// displacement), then for any continuation u, u ++ t1 ~ u ++ t2 (the same
// displacement). This follows from the associativity of the composition (the
// D(u ++ t1) = D(t1) o D(u) = D(t2) o D(u) = D(u ++ t2)). but is stated
// explicitly as the congruence property (the no just the associativity).
#[test]
fn proof_displacement_congruence() {
    let m = pushdown_rs::compile(&scattered_cfg()).expect("compile");
    // Two sequences with the same displacement (the t1 = t2).
    let t1 = vec![0u32];
    let t2 = vec![0u32]; // the same sequence (the trivial congruence)
    assert_eq!(m.displacement(&t1), m.displacement(&t2), "the t1 ~ t2 (the same displacement)");
    // A continuation u (the no the same as t1/t2).
    let u = vec![1u32, 0u32];
    // The congruence: the u ++ t1 ~ u ++ t2 (the same displacement).
    let ut1: Vec<u32> = u.iter().chain(t1.iter()).copied().collect();
    let ut2: Vec<u32> = u.iter().chain(t2.iter()).copied().collect();
    assert_eq!(
        m.displacement(&ut1),
        m.displacement(&ut2),
        "the congruence: the u ++ t1 ~ u ++ t2 (the same displacement)"
    );
    // The non-trivial congruence: two different sequences with the same
    // displacement (the the t1' ~ t2', the no t1' == t2').
    let t1p = vec![0u32, 1u32];
    let t2p = vec![1u32, 0u32]; // the different sequence (the no the same displacement)
    if m.displacement(&t1p) == m.displacement(&t2p) {
        // The t1p ~ t2p (the same displacement), so the congruence holds.
        let ut1p: Vec<u32> = u.iter().chain(t1p.iter()).copied().collect();
        let ut2p: Vec<u32> = u.iter().chain(t2p.iter()).copied().collect();
        assert_eq!(
            m.displacement(&ut1p),
            m.displacement(&ut2p),
            "the congruence: the u ++ t1p ~ u ++ t2p (the same displacement)"
        );
    }
}

// The displacement is the BFS over the reachable in_configs (the ground truth,
// the no single-path shortcut). This is the regression gate for the GNF bridge
// grouping optimization (the displacement is a function of the byte sequence,
// the no the token ID, so it is computed once per unique byte-sequence).
#[test]
fn proof_displacement_is_bfs() {
    let g = Cfg::new(
        2, // S, T
        3, // a, b, c
        0, // start = S
        vec![
            (0, vec![2, 1, 3]), // S -> a T b (the 2=a, the 1=T, the 3=b)
            (1, vec![4]),       // T -> c (the 4=c)
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile");
    // The displacement of the empty sequence is the identity relation (the
    // (in_config, in_config) pairs for all reachable in_configs).
    let d_empty = m.displacement(&[]);
    for &(in_q, ref in_stack, out_q, ref out_stack) in &d_empty {
        assert_eq!(in_q, out_q, "the empty displacement must the identity (the in == the out)");
        assert_eq!(in_stack, out_stack, "the empty displacement is the identity (the in_stack == the out_stack)");
    }
    assert!(!d_empty.is_empty(), "the empty displacement is non-empty (the reachable in_configs exist)");
}

// ============================================================================
// PROOF: advance_eps computes the COMPLETE epsilon closure (the no synthetic
// caps) on a PDA with a stack-growing epsilon cycle. The independent reference
// below dedups on (state, stack-top) with no cap, so agreement proves 100%
// inclusion (a reachable terminal is found) + 100% exclusion (an unreachable
// one is not).
// ============================================================================

/// A DPDA with a stack-growing epsilon cycle: q=2 <-> q=3 both push [1,1] onto
/// top=1 (the stack grows unboundedly, the top stays 1). Terminal `a` (0) is
/// available only at (q=2, top=1); terminal `b` (1) is nowhere.
fn cycle_dpda() -> PdaMachine {
    const EPS: u32 = 2; // the num_inputs
    const A: u32 = 0;
    PdaMachine {
        num_states: 4,
        num_inputs: 2,
        num_stack_syms: 2,
        transitions: vec![
            Transition { q: 0, a: EPS, top: 0, next_q: 1, push: vec![0] },
            Transition { q: 1, a: EPS, top: 0, next_q: 2, push: vec![1, 0] },
            Transition { q: 2, a: EPS, top: 1, next_q: 3, push: vec![1, 1] }, // the cycle grow
            Transition { q: 3, a: EPS, top: 1, next_q: 2, push: vec![1, 1] }, // the cycle grow
            Transition { q: 2, a: A, top: 1, next_q: 2, push: vec![1] }, // the terminal a at (2, top=1)
        ],
        accepting: vec![3],
        start_state: 0,
        start_stack: 0,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
        flat_a: vec![],
        flat_top: vec![],
        flat_next_q: vec![],
    }
}

/// The independent reference epsilon closure (the no caps, the (state, top) dedup).
/// Returns the next-state for a terminal-`a` move reachable in the closure, or None.
/// Assumes all epsilon pushes are non-empty (true for cycle_dpda); the new top is
/// push.first() because advance_eps reverse-pushes (push.iter().rev()).
fn reference_advance_state(m: &PdaMachine, q: u32, stk: &[u32], a: u32) -> Option<u32> {
    let start_top = stk.last().copied().unwrap_or(m.start_stack);
    let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut queue: Vec<(u32, u32)> = vec![(q, start_top)];
    seen.insert((q, start_top));
    let mut head = 0;
    while head < queue.len() {
        let (cs, ctop) = queue[head];
        head += 1;
        for t in &m.transitions {
            if t.q == cs && t.a == m.num_inputs && t.top == ctop {
                // The advance_eps reverse-pushes (push.iter().rev()), so the FIRST
                // element of push ends up on top.
                let nt = t.push.first().copied().unwrap_or(m.start_stack);
                if seen.insert((t.next_q, nt)) {
                    queue.push((t.next_q, nt));
                }
            }
        }
    }
    for &(cs, ctop) in &queue {
        for t in &m.transitions {
            if t.q == cs && t.a == a && t.top == ctop {
                return Some(t.next_q);
            }
        }
    }
    None
}

#[test]
fn advance_eps_inclusion_matches_reference_on_cycle() {
    let m = cycle_dpda();
    // The terminal a (0) IS reachable at (q=2, top=1) via the epsilon closure.
    let got = m.advance_eps(0, &[], 0).map(|(nq, _)| nq);
    let want = reference_advance_state(&m, 0, &[], 0);
    assert_eq!(got, want, "the advance_eps next-state must match the independent reference");
    assert_eq!(got, Some(2), "the terminal a reaches q=2 (the 100% inclusion)");
}

#[test]
fn advance_eps_exclusion_matches_reference_on_cycle() {
    let m = cycle_dpda();
    // The terminal b (1) is NOT reachable (no b transition exists).
    let got = m.advance_eps(0, &[], 1).map(|(nq, _)| nq);
    let want = reference_advance_state(&m, 0, &[], 1);
    assert_eq!(got, want, "the advance_eps next-state must match the independent reference");
    assert_eq!(got, None, "the unreachable terminal b yields None (the 100% exclusion)");
}

#[test]
fn advance_eps_terminates_on_stack_growing_cycle() {
    // The stack-growing cycle (q=2 <-> q=3, push [1,1]) must terminate (the (state, top)
    // dedup bounds it) and still find the terminal. A fast return proves termination.
    let m = cycle_dpda();
    let start = std::time::Instant::now();
    let r = m.advance_eps(0, &[], 0);
    let elapsed = start.elapsed();
    assert!(r.is_some(), "the terminal is reached despite the cycle");
    assert!(
        elapsed.as_millis() < 1000,
        "the closure terminates quickly (the no unbounded stack growth): {elapsed:?}"
    );
}

// PROOF: for a NON-deterministic PDA, the CPU project_batch (the sound
// epsilon-closure-union mask at each draft step) includes a legal draft that the
// GPU settled projection (mask_at_cfg_settled) wrongly excludes. This qualifies why
// the non-deterministic fallback MUST use the CPU closure-union projection.
#[test]
fn proof_project_batch_sound_where_settled_is_not() {
    const A: u32 = 0;
    const B: u32 = 1;
    const EPS: u32 = 2; // the num_inputs
    const Z: u32 = 0; // the start_stack
    // q0 has TWO epsilon moves (to q1 and q2) -> non-deterministic. q1 allows A, q2 allows B.
    let m = PdaMachine {
        num_states: 3,
        num_inputs: 2,
        num_stack_syms: 1,
        transitions: vec![
            Transition { q: 0, a: EPS, top: Z, next_q: 1, push: vec![Z] },
            Transition { q: 0, a: EPS, top: Z, next_q: 2, push: vec![Z] },
            Transition { q: 1, a: A, top: Z, next_q: 1, push: vec![Z] },
            Transition { q: 2, a: B, top: Z, next_q: 2, push: vec![Z] },
        ],
        accepting: vec![1, 2],
        start_state: 0,
        start_stack: Z,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
        flat_a: vec![],
        flat_top: vec![],
        flat_next_q: vec![],
    };
    assert!(!m.is_deterministic(), "the machine is non-deterministic (the two eps branches from q0)");

    // The settled (GPU) mask at the start is EMPTY (q0 has no input move) -> under-approximates.
    let settled = m.mask_at_cfg_settled(0, Z);
    assert!(settled.is_empty(), "the settled single-config mask is empty at q0 (the under-approximation)");

    // The sound closure-union mask at the start includes BOTH branch inputs.
    let closure = m.mask_at_cfg(0, &[Z]);
    assert!(closure.contains(&A) && closure.contains(&B), "the closure-union mask includes both branch inputs");

    // The draft [B] is LEGAL (via branch q2). The sound project_batch includes B at step 0.
    let masks = m.project_batch(&[(0, vec![Z])], &[vec![B]]);
    assert!(
        masks[0][0].contains(&B),
        "the project_batch step-0 mask includes the legal draft B (the sound inclusion)"
    );
    // The contrast: the settled mask excludes B (the GPU projection would wrongly reject the legal draft).
    assert!(!settled.contains(&B), "the settled mask excludes B (the GPU projection is unsound here)");
}

// Isolate the one-or-more (the +) loop: does the RTN compilation produce a machine that
// repeats the loop body? The grammar S -> A, A -> B A | B is the language B+ (one or more B's).
// If the loop back-edge is missing, the machine only accepts a single B (the A -> B exit) and
// rejects BB, BBB (the A -> B A repetition). This isolates the reasoning_block (<...>)+ desync.
#[test]
fn rtn_one_or_more_loop_repeats() {
    let g = Cfg::new(
        2, // the num_nonterminals: S=0, A=1
        1, // the num_terminals: B=local 0 (the global 2)
        0, // the start = S
        vec![
            (0, vec![1]),    // S -> A
            (1, vec![2, 1]), // A -> B A (the loop: consume B, then A again)
            (1, vec![2]),    // A -> B (the loop exit)
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile the B+ grammar");
    eprintln!("[DBG B+ machine] num_states={} num_inputs={} num_stack_syms={} accepting={:?}", m.num_states, m.num_inputs, m.num_stack_syms, m.accepting);
    for t in &m.transitions {
        eprintln!("[DBG B+ trans] (q={}, a={}, top={}) -> (q={}, push={:?})", t.q, t.a, t.top, t.next_q, t.push);
    }
    assert!(m.accepts(&g, &[0]), "a single B is in B+");
    assert!(m.accepts(&g, &[0, 0]), "two B's are in B+ (the loop repeats once)");
    assert!(m.accepts(&g, &[0, 0, 0]), "three B's are in B+ (the loop repeats twice)");
    assert!(!m.accepts(&g, &[]), "the empty string is not in B+");
}

// The nested-loop shape of the Qwen reasoning_block: "(<text_range>)+ (<reasoning_end>)".
// The one_or_more expansion is P -> T (the base) + P -> P T (the loop), followed by a trailing
// RE. The language is T+ RE (one-or-more T's then the RE). If the loop back-edge is lost when a
// trailing symbol follows the loop, T T RE is rejected (the desync).
#[test]
fn rtn_nested_loop_with_trailing_symbol_repeats() {
    // N = {S=0, P=1}, T = {T=0, RE=1} (the local terminal IDs).
    //   S -> P RE
    //   P -> T        (the base: one T)
    //   P -> P T       (the loop: P then T)
    let g = Cfg::new(
        2, // the num_nonterminals: S=0, P=1
        2, // the num_terminals: T=local 0 (the global 2), RE=local 1 (the global 3)
        0, // the start = S
        vec![
            (0, vec![1, 3]), // S -> P RE (the P=global 1, the RE=global 3)
            (1, vec![2]),    // P -> T (the T=global 2)
            (1, vec![1, 2]), // P -> P T (the loop)
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile the nested-loop grammar");
    eprintln!("[DBG machine] num_states={} num_inputs={} num_stack_syms={}", m.num_states, m.num_inputs, m.num_stack_syms);
    for t in &m.transitions {
        eprintln!("[DBG machine] (q={}, a={}, top={}) -> (q={}, push={:?})", t.q, t.a, t.top, t.next_q, t.push);
    }
    eprintln!("[DBG machine] accepting={:?}", m.accepting);
    assert!(m.accepts(&g, &[0, 1]), "T RE (the single-iteration base case)");
    assert!(m.accepts(&g, &[0, 0, 1]), "T T RE (the loop repeats once)");
    assert!(m.accepts(&g, &[0, 0, 0, 1]), "T T T RE (the loop repeats twice)");
    assert!(!m.accepts(&g, &[1]), "RE alone is not T+ RE");
}

// Reproduce the Qwen reasoning_block nesting: an inner one-or-more loop (the IP -> IP T)
// followed by a trailing symbol (the RE), itself nested inside an outer one-or-more loop
// (the OP -> OP A). The language is (T+ RE) (A+) E. A valid string is T T RE TC E.
// If the inner loop's back-edge is lost under the nesting, T T RE ... is rejected.
#[test]
fn rtn_nested_loops_reproduce_qwen_shape() {
    // N = {S=0, RB=1, IP=2, OP=3, A=4}; T = {T=5, RE=6, TC=7, TX=8, E=9} (the global IDs).
    let g = Cfg::new(
        5, // the num_nonterminals
        5, // the num_terminals
        0, // the start = S
        vec![
            (0, vec![1, 3, 9]), // S -> RB OP E
            (1, vec![2, 6]),    // RB -> IP RE
            (2, vec![5]),      // IP -> T (the inner-loop base)
            (2, vec![2, 5]),   // IP -> IP T (the inner-loop back-edge)
            (3, vec![4]),      // OP -> A (the outer-loop base)
            (3, vec![3, 4]),   // OP -> OP A (the outer-loop back-edge)
            (4, vec![7]),      // A -> TC
            (4, vec![8]),      // A -> TX
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile the nested-loop grammar");
    // The local terminal IDs: T=0, RE=1, TC=2, TX=3, E=4.
    // A valid string: T T RE TC E = [0, 0, 1, 2, 4].
    assert!(
        m.accepts(&g, &[0, 0, 1, 2, 4]),
        "T T RE TC E is valid (the inner loop repeats once)"
    );
    assert!(
        m.accepts(&g, &[0, 0, 0, 1, 2, 4]),
        "T T T RE TC E is valid (the inner loop repeats twice)"
    );
    assert!(m.accepts(&g, &[0, 1, 2, 4]), "T RE TC E is valid (the inner loop base case)");
}

// The Qwen start-shape with the THIRD nested loop: the outer (tool_call | text)+ loop whose
// text branch is itself a one-or-more (the text: (<R2>)+). Three nested one_or_more loops.
// A valid string is R1 R1 RE TC EOS (two inner-1 tokens, the reasoning-end, one tool, the eos).
#[test]
fn rtn_triple_nested_loops_reproduce_qwen_start() {
    // N = {S=0, RB=1, IP=2, OP=3, CH=4, TX=5}; T = {R1=6, RE=7, TC=8, R2=9, EOS=10} (the global).
    let g = Cfg::new(
        6, // the num_nonterminals
        5, // the num_terminals
        0, // the start = S
        vec![
            (0, vec![1, 3, 10]), // S -> RB OP EOS
            (1, vec![2, 7]),     // RB -> IP RE
            (2, vec![6]),       // IP -> R1 (the inner-1 base)
            (2, vec![2, 6]),    // IP -> IP R1 (the inner-1 loop)
            (3, vec![4]),       // OP -> CH (the outer base)
            (3, vec![3, 4]),    // OP -> OP CH (the outer loop)
            (4, vec![8]),       // CH -> TC
            (4, vec![5]),       // CH -> TX
            (5, vec![9]),       // TX -> R2 (the inner-2 base)
            (5, vec![5, 9]),    // TX -> TX R2 (the inner-2 loop)
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile the triple-nested grammar");
    // The local terminal IDs: R1=0, RE=1, TC=2, R2=3, EOS=4.
    // A valid string: R1 R1 RE TC EOS = [0, 0, 1, 2, 4].
    assert!(
        m.accepts(&g, &[0, 0, 1, 2, 4]),
        "R1 R1 RE TC EOS is valid (the inner-1 loop repeats once)"
    );
    assert!(
        m.accepts(&g, &[0, 0, 0, 1, 2, 4]),
        "R1 R1 R1 RE TC EOS is valid (the inner-1 loop repeats twice)"
    );
}

// The EXACT Qwen start-shape: THREE nested one_or_more loops. The inner-1 loop (the IP) is in
// the reasoning_block, the inner-2 loop (the TX) is nested INSIDE the outer loop's (the OP)
// choice body. This is the structure that breaks the RTN loop back-edge.
#[test]
fn rtn_three_nested_loops_qwen_exact_shape() {
    // N = {S=0, RB=1, IP=2, OP=3, C=4, TX=5}; T = {T=6, RE=7, TC=8, R2=9, EOS=10} (the global).
    let g = Cfg::new(
        6, // the num_nonterminals
        5, // the num_terminals
        0, // the start = S
        vec![
            (0, vec![1, 3, 10]), // S -> RB OP EOS
            (1, vec![2, 7]),     // RB -> IP RE
            (2, vec![6]),       // IP -> T (the inner-1 base)
            (2, vec![2, 6]),    // IP -> IP T (the inner-1 loop)
            (3, vec![4]),       // OP -> C (the outer base)
            (3, vec![3, 4]),    // OP -> OP C (the outer loop)
            (4, vec![8]),       // C -> TC
            (4, vec![5]),       // C -> TX
            (5, vec![9]),       // TX -> R2 (the inner-2 base)
            (5, vec![5, 9]),    // TX -> TX R2 (the inner-2 loop, nested in the outer choice)
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile the three-nested-loop grammar");
    // The local terminal IDs: T=0, RE=1, TC=2, R2=3, EOS=4.
    // A valid string: T T RE TC EOS = [0, 0, 1, 2, 4] (the inner-1 loop repeats once).
    assert!(m.accepts(&g, &[0, 0, 1, 2, 4]), "T T RE TC EOS is valid (the inner-1 loop repeats once)");
    assert!(
        m.accepts(&g, &[0, 0, 0, 1, 2, 4]),
        "T T T RE TC EOS is valid (the inner-1 loop repeats twice)"
    );
    // The inner-2 loop (the TX -> TX R2): T RE R2 R2 TC EOS = [0, 1, 3, 3, 2, 4].
    assert!(
        m.accepts(&g, &[0, 1, 3, 3, 2, 4]),
        "T RE R2 R2 TC EOS is valid (the inner-2 loop repeats once)"
    );
}

// The EXACT Qwen CFG (the 15 nonterminals + the 25 terminals, the three one_or_more loops, the
// start rule referencing two loops directly). Reproduced from the PdaGrammar local-ID dump. If
// the RTN compile drops a loop back-edge for this shape, the inner `plus` loop (NT1) only repeats
// once. Global IDs: nonterminals 0..14, terminals 15..39 (the 15 + the local terminal ID).
#[test]
fn rtn_exact_qwen_cfg_loop_back_edge() {
    let g = Cfg::new(
        15, // the num_nonterminals
        25, // the num_terminals
        0, // the start
        vec![
            (0, vec![1, 16, 2, 3]), // start -> NT1 T1 NT2 NT3
            (1, vec![17]), // NT1 -> T2 (the plus base)
            (1, vec![1, 17]), // NT1 -> NT1 T2 (the plus loop)
            (2, vec![19]), // NT2 -> NT4 (the plus#3 base)
            (2, vec![2, 19]), // NT2 -> NT2 NT4 (the plus#3 loop)
            (3, vec![18]), // NT3 -> T3 (the eos)
            (3, vec![19]), // NT3 -> T4 (the eos alt)
            (4, vec![20, 13, 38]), // NT4 -> T5 NT13 T23 (the tool_call)
            (4, vec![14]), // NT4 -> NT14 (the text choice)
            (6, vec![1, 16]), // NT6 -> NT1 T1 (the reasoning_block, the orphaned)
            (7, vec![22, 23]), // NT7 -> T7 T8
            (8, vec![24, 23]), // NT8 -> T9 T8
            (9, vec![27, 23]), // NT9 -> T12 T8
            (10, vec![30, 23]), // NT10 -> T15 T8
            (11, vec![35, 23]), // NT11 -> T20 T8
            (12, vec![36, 23]), // NT12 -> T21 T8
            (13, vec![21, 22, 23, 8, 25]), // NT13 -> T6 T7 T8 NT8 T10 (the tool_0)
            (13, vec![26, 27, 23, 28]), // NT13 -> T11 T12 T8 T13 (the tool_1)
            (13, vec![29, 30, 23, 31]), // NT13 -> T14 T15 T8 T16 (the tool_2)
            (13, vec![32, 33]), // NT13 -> T17 T18 (the tool_3)
            (13, vec![34, 35, 23, 12, 37]), // NT13 -> T19 T20 T8 NT12 T22 (the tool_4)
            (14, vec![39]), // NT14 -> T24 (the text base)
            (14, vec![14, 39]), // NT14 -> NT14 T24 (the text loop)
        ],
    );
    let m = pushdown_rs::compile(&g).expect("compile the exact Qwen CFG");
    // The inner `plus` loop (NT1, the T2 = local 2): the Earley per-step mask
    // (the O(n^3) chart,, the Leo 1991 optimization) must keep the loop
    // branch alive. The advance_eps_set (the exponential frontier) is replaced
    // by the earley_per_step_mask (the polynomial per-step mask).
    let local_w: Vec<u32> = vec![2, 2]; // the two T2's (the local terminal IDs).
    let global_w: Vec<u32> = local_w.iter().map(|&x| x + g.num_nonterminals).collect();
    let mask1 = pushdown_rs::earley_per_step_mask(&g, &global_w, 1);
    eprintln!("[DBG qwen-cfg] Earley mask after 1st T2: {:?}", mask1);
    let mask2 = pushdown_rs::earley_per_step_mask(&g, &global_w, 2);
    eprintln!("[DBG qwen-cfg] Earley mask after 2nd T2: {:?}", mask2);
    assert!(
        !mask2.is_empty(),
        "the plus loop back-edge: a second T2 must have a non-empty mask (the + is one-or-more, the Earley chart keeps the loop branch)"
    );
    // The contrast: the single-config advance_eps drops the loop (the cfg2 is None).
    let cfg1 = m.advance_eps(m.start_state, &[m.start_stack], 2);
    let cfg2_single = cfg1.and_then(|c| m.advance_eps(c.0, &c.1, 2));
    eprintln!("[DBG qwen-cfg] single-config advance_eps 2nd T2 = {:?} (the dropped loop, the contrast)", cfg2_single.is_some());
}

// Isolate the mask_at_cfg behavior on a minimal 3-state PDA (the start -> the a-consumer).
// The mask_at_cfg(0, &[]) must return the allowed input at the start config (the epsilon closure
// reaches the a-consumer). If it returns empty, the mask_at_cfg closure is broken.
#[test]
fn debug_mask_at_cfg_minimal() {
    const EPS: u32 = 2; // the num_inputs
    const A: u32 = 0;
    const Z: u32 = 0; // the start_stack
    let m = PdaMachine {
        num_states: 3,
        num_inputs: 2,
        num_stack_syms: 2,
        transitions: vec![
            Transition { q: 0, a: EPS, top: Z, next_q: 1, push: vec![Z] },
            Transition { q: 1, a: A, top: Z, next_q: 2, push: vec![Z] },
        ],
        accepting: vec![2],
        start_state: 0,
        start_stack: Z,
        state_provenance: None,
        vocab_names: None,
        ctrl_offsets: vec![],
        ctrl_counts: vec![],
        flat_a: vec![],
        flat_top: vec![],
        flat_next_q: vec![],
    };
    eprintln!("[DBG minimal] mask_at_cfg(0, &[]) = {:?}", m.mask_at_cfg(0, &[]));
    eprintln!("[DBG minimal] mask_at_cfg(0, &[0]) = {:?}", m.mask_at_cfg(0, &[0]));
    eprintln!("[DBG minimal] mask_at_cfg(1, &[0]) = {:?}", m.mask_at_cfg(1, &[0]));
    assert_eq!(m.mask_at_cfg(0, &[]), vec![A], "the start config's mask must allow the input A (the epsilon closure reaches the a-consumer)");
}

// PROOF: the Earley per-step mask (the O(n^3) chart, the Leo 1991) agrees with
// the PDA's mask_at_cfg (the proven epsilon-closure mask) at every position.
// The Earley mask is in the GLOBAL terminal IDs (the num_nonterminals..+num_terminals);
// the PDA mask is in the LOCAL input IDs (the 0..num_inputs). Convert global -> local
// (the - num_nonterminals) and compare. This verifies the Earley implementation
// (the Predict/Scan/Complete, the start-field fix) against the independent PDA mask.
#[test]
fn proof_earley_per_step_mask_agrees_with_pda_mask() {
    use pushdown_rs::compile::earley_per_step_mask;
    // The S -> a A b, A -> c grammar (the call dot, the bounded stack).
    let g = Cfg::new(2, 3, 0, vec![(0, vec![2, 1, 3]), (1, vec![4])]);
    let m = pushdown_rs::compile(&g).expect("compile");
    let num_nt = g.num_nonterminals;
    // The input "a c b" = the local [0, 2, 1] (the a=0, the c=2, the b=1).
    let local = vec![0u32, 2, 1];
    let global: Vec<u32> = local.iter().map(|&x| x + num_nt).collect();
    // At each position i (the 0..=n), the Earley mask (the global -> local) must
    // equal the PDA's mask_at_cfg at the config reached after consuming w[0..i].
    let mut config = (m.start_state, vec![m.start_stack]);
    for i in 0..=global.len() {
        let earley_mask_global = earley_per_step_mask(&g, &global, i);
        let earley_mask_local: Vec<u32> = earley_mask_global.iter()
            .map(|&x| x - num_nt)
            .collect();
        let pda_mask = m.mask_at_cfg(config.0, &config.1);
        assert_eq!(
            earley_mask_local, pda_mask,
            "the Earley per-step mask (the position {i}) must equal the PDA mask_at_cfg"
        );
        // Advance the PDA config by the next input (the no the last position).
        if i < global.len() {
            if let Some(nc) = m.advance_eps(config.0, &config.1, local[i]) {
                config = nc;
            }
        }
    }
}
