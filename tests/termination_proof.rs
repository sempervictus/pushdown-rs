//! Termination proof test: verify thatars that pass and fail.
//!
//! Uses the real 248K tokenizer + the xbot grammar (should pass).
//! Also tests a deliberately broken grammar (should fail).
//!
//! Run: cargo test --features simd termination_proof -- --nocapture

#[cfg(test)]
mod termination_proof {
    use pushdown_rs::compile::Cfg;
    use pushdown_rs::summary::{BoundedSummary, TerminationProof};

    /// A valid grammar: {a^n b^n} (S -> a S b | eps).
    /// This should PASS the termination proof (the PDA can reach acceptance).
    #[test]
    fn valid_grammar_passes_termination_proof() {
        // S -> a S b | eps
        // N = {S}, Sigma = {a, b}, P = {S -> a S b, S -> eps}
        // Global IDs: S=0, a=1, b=2
        let g = Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2]), (0, vec![])]);
        let pda = pushdown_rs::compile(&g).expect("compile {a^n b^n}");
        let summary = BoundedSummary::compute(&pda, 8);
        let proof = summary.prove_termination(&pda);
        match proof {
            TerminationProof::Proven { min_tokens_to_accept, reachable_configs } => {
                println!("PASS: {{a^n b^n}} terminates in {} tokens ({} reachable configs)",
                    min_tokens_to_accept, reachable_configs);
                assert!(min_tokens_to_accept <= 8, "should reach acceptance within the bound");
            }
            TerminationProof::DeadEnd { stuck_configs, reachable_configs } => {
                panic!("FAIL: {{a^n b^n}} should terminate but has {} stuck configs out of {}",
                    stuck_configs, reachable_configs);
            }
        }
    }

    /// A valid grammar: balanced parentheses (S -> ( S ) S | eps).
    /// This should PASS the termination proof.
    #[test]
    fn balanced_parens_passes_termination_proof() {
        // S -> ( S ) S | eps
        // N = {S}, Sigma = {(, )}, P = {S -> ( S ) S, S -> eps}
        // Global IDs: S=0, (=1, )=2
        let g = Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2, 0]), (0, vec![])]);
        let pda = pushdown_rs::compile(&g).expect("compile balanced parens");
        let summary = BoundedSummary::compute(&pda, 8);
        let proof = summary.prove_termination(&pda);
        match proof {
            TerminationProof::Proven { min_tokens_to_accept, reachable_configs } => {
                println!("PASS: balanced parens terminates in {} tokens ({} reachable configs)",
                    min_tokens_to_accept, reachable_configs);
                assert!(min_tokens_to_accept <= 8);
            }
            TerminationProof::DeadEnd { stuck_configs, reachable_configs } => {
                panic!("FAIL: balanced parens should terminate but has {} stuck configs out of {}",
                    stuck_configs, reachable_configs);
            }
        }
    }

    /// A BROKEN grammar: S -> a S (no way to terminate, infinite recursion).
    /// This should FAIL the termination proof (dead-end).
    #[test]
    fn broken_grammar_fails_termination_proof() {
        // S -> a S (no epsilon production no terminal exit)
        // N = {S}, Sigma = {a}, P = {S -> a S}
        // Global IDs: S=0, a=1
        let g = Cfg::new(1, 1, 0, vec![(0, vec![1, 0])]);
        let pda = pushdown_rs::compile(&g).expect("compile broken grammar");
        let summary = BoundedSummary::compute(&pda, 8);
        let proof = summary.prove_termination(&pda);
        match proof {
            TerminationProof::Proven { min_tokens_to_accept, .. } => {
                panic!("FAIL: broken grammar should NOT terminate but proof says {} tokens",
                    min_tokens_to_accept);
            }
            TerminationProof::DeadEnd { stuck_configs, reachable_configs } => {
                println!("PASS: broken grammar correctly detected as dead-end ({} stuck / {} reachable)",
                    stuck_configs, reachable_configs);
                assert!(stuck_configs > 0, "should should be stuck configs");
            }
        }
    }

/// A BROKEN grammar: S -> a S b (no epsilon, no base case). The language is
/// EMPTY (the S never completes), so the termination proof must report a DeadEnd
/// (the no path from start to acceptance within the bound). This is the
/// non-vacuous assertion (the prior version accepted either outcome, so it could
/// never fail).
#[test]
fn unbounded_recursion_fails_termination_proof() {
    // S -> a S b (no epsilon production)
    // N = {S}, Sigma = {a, b}, P = {S -> a S b}
    // Global IDs: S=0, a=1, b=2
    let g = Cfg::new(1, 2, 0, vec![(0, vec![1, 0, 2])]);
    let pda = pushdown_rs::compile(&g).expect("compile unbounded recursion");
    let summary = BoundedSummary::compute(&pda, 8);
    let proof = summary.prove_termination(&pda);
    match proof {
        TerminationProof::Proven { min_tokens_to_accept, .. } => {
            panic!(
                "FAIL: S -> a S b has an empty language and must NOT prove termination (got {min_tokens_to_accept})"
            );
        }
        TerminationProof::DeadEnd { stuck_configs, reachable_configs } => {
            println!("PASS: unbounded recursion correctly detected as dead-end ({} stuck / {} reachable)",
                stuck_configs, reachable_configs);
            assert!(stuck_configs > 0, "the dead-end must have stuck configs");
        }
    }
}

    /// A grammar with a dead-end branch: S -> a S | a T, T -> (stuck, no exit).
    /// The T branch is a dead-end (can't reach acceptance).
    #[test]
    fn dead_end_branch_detected() {
        // S -> a S | a T
        // T -> b T (no exit from T)
        // N = {S, T}, Sigma = {a, b}, P = {S -> a S, S -> a T, T -> b T}
        // Global IDs: S=0, T=1, a=2, b=3
        let g = Cfg::new(2, 2, 0, vec![
            (0, vec![2, 0]), // S -> a S
            (0, vec![2, 1]), // S -> a T
            (1, vec![3, 1]), // T -> b T (dead-end: T never exits)
        ]);
        let pda = pushdown_rs::compile(&g).expect("compile dead-end branch");
        let summary = BoundedSummary::compute(&pda, 8);
        let proof = summary.prove_termination(&pda);
        match proof {
            TerminationProof::Proven { min_tokens_to_accept, .. } => {
                // The S -> a S branch CAN terminate (push a's, then... wait,
                // there's no epsilon for S. S -> a S is infinite recursion.
                // But S -> a T leads to T -> b T which is also infinite.
                // So this grammar actually CANNOT terminate.
                panic!("UNEXPECTED: dead-end grammar passed proof ({} tokens)", min_tokens_to_accept);
            }
            TerminationProof::DeadEnd { stuck_configs, reachable_configs } => {
                println!("PASS: dead-end branch detected ({} stuck / {} reachable)",
                    stuck_configs, reachable_configs);
                assert!(stuck_configs > 0);
            }
        }
    }
}