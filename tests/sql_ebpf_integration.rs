//! The SQL + the eBPF integration proofs (the the exhaustive differential).
//!
//! Proves the PDA (the RTN compilation of the CFG) matches:
//! - the core's independent `cfg_accepts` oracle (the the math definition, the
//!   zero shared code with the PDA) EXHAUSTIVELY over the bounded terminal
//!   alphabet (the the inclusive + the exclusive construction proof);
//! - the independent-register oracles (the the recursive-descent SQL parser, the
//!   the call-depth eBPF walker) on the curated boundary cases (the the language
//!   correctness proof).

use pushdown_rs::compile::{Cfg, kappa};
use pushdown_rs::machine::PdaMachine;
use pushdown_rs::oracle::cfg_accepts;
use pushdown_rs::pda::Npda;

// ==================== the SQL subset (the the left-factored, the the DCFL) ====================

// The local terminal ids (the the 0..9).
const SELECT: u32 = 0;
const FROM: u32 = 1;
const WHERE: u32 = 2;
const OR: u32 = 3;
const STAR: u32 = 4;
const LPAREN: u32 = 5;
const RPAREN: u32 = 6;
const IDENT: u32 = 7;
const NUM: u32 = 8;

fn sql_cfg() -> Cfg {
    Cfg::new(
        8, // the N = {Stmt, Select, WhereOpt, Expr, ExprTail, Term, TermTail, Factor}
        9, // the Sigma
        0, // the S = Stmt
        vec![
            (0, vec![1]), // Stmt -> Select
            (1, vec![8, 3, 9, 15, 2]), // Select -> SELECT Expr FROM IDENT WhereOpt
            (2, vec![10, 3]), // WhereOpt -> WHERE Expr
            (2, vec![]), // WhereOpt -> eps
            (3, vec![5, 4]), // Expr -> Term ExprTail
            (4, vec![11, 5, 4]), // ExprTail -> OR Term ExprTail
            (4, vec![]), // ExprTail -> eps
            (5, vec![7, 6]), // Term -> Factor TermTail
            (6, vec![12, 7, 6]), // TermTail -> * Factor TermTail
            (6, vec![]), // TermTail -> eps
            (7, vec![13, 3, 14]), // Factor -> ( Expr )
            (7, vec![15]), // Factor -> IDENT
            (7, vec![16]), // Factor -> NUM
        ],
    )
}

/// The recursive-descent SQL oracle (the the independent register, the the zero
/// shared code with the PDA).
struct SqlParser<'a> {
    toks: &'a [u32],
    pos: usize,
}

impl<'a> SqlParser<'a> {
    fn peek(&self) -> Option<u32> {
        self.toks.get(self.pos).copied()
    }
    fn expect(&mut self, t: u32) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn parse_factor(&mut self) -> bool {
        match self.peek() {
            Some(LPAREN) => {
                self.pos += 1;
                self.parse_expr() && self.expect(RPAREN)
            }
            Some(IDENT) | Some(NUM) => {
                self.pos += 1;
                true
            }
            _ => false,
        }
    }
    fn parse_term(&mut self) -> bool {
        if !self.parse_factor() {
            return false;
        }
        while self.peek() == Some(STAR) {
            self.pos += 1;
            if !self.parse_factor() {
                return false;
            }
        }
        true
    }
    fn parse_expr(&mut self) -> bool {
        if !self.parse_term() {
            return false;
        }
        while self.peek() == Some(OR) {
            self.pos += 1;
            if !self.parse_term() {
                return false;
            }
        }
        true
    }
    fn parse_whereopt(&mut self) -> bool {
        if self.peek() == Some(WHERE) {
            self.pos += 1;
            self.parse_expr()
        } else {
            true
        }
    }
    fn parse_select(&mut self) -> bool {
        self.expect(SELECT)
            && self.parse_expr()
            && self.expect(FROM)
            && self.expect(IDENT)
            && self.parse_whereopt()
    }
    fn accepts(&mut self) -> bool {
        let ok = self.parse_select();
        ok && self.pos == self.toks.len()
    }
}

fn sql_oracle(toks: &[u32]) -> bool {
    SqlParser { toks, pos: 0 }.accepts()
}

// ==================== the eBPF program (the the well-nested call/return, the the DCFL) ====================

// The local terminal ids (the the 0..4).
const CALL: u32 = 0;
const RET: u32 = 1;
const OP: u32 = 2;
const EXIT: u32 = 3;

fn ebpf_cfg() -> Cfg {
    Cfg::new(
        3, // the N = {Prog, Sub, Unit}
        4, // the Sigma = {CALL, RET, OP, EXIT}
        0, // the S = Prog
        vec![
            (0, vec![1, 6]), // Prog -> Sub EXIT
            (1, vec![2, 1]), // Sub -> Unit Sub
            (1, vec![]), // Sub -> eps
            (2, vec![5]), // Unit -> OP
            (2, vec![3, 1, 4]), // Unit -> CALL Sub RET
        ],
    )
}

/// The call-depth eBPF walker (the the independent register).
fn ebpf_walker(tokens: &[u32]) -> bool {
    let is_sub = |prefix: &[u32]| -> bool {
        let mut depth: i32 = 0;
        for &t in prefix {
            match t {
                CALL => depth += 1,
                OP => {}
                RET => {
                    if depth == 0 {
                        return false;
                    }
                    depth -= 1;
                }
                _ => return false,
            }
        }
        depth == 0
    };
    match tokens.last() {
        Some(&EXIT) => is_sub(&tokens[..tokens.len() - 1]),
        _ => false,
    }
}

// ==================== the exhaustive differential (the the PDA == the cfg_accepts) ====================

/// Generate all sequences over the global terminal alphabet (the the num_nt..
/// num_nt+num_tm) of length 0..=max_len, and assert the PDA == the cfg_accepts
/// oracle on every one (the the inclusive + the exclusive construction proof).
fn exhaustive_differential(m: &PdaMachine, g: &Cfg, max_len: usize) {
    let num_nt = g.num_nonterminals; // the u32
    let num_tm_us = g.num_terminals as usize;
    let mut checked = 0usize;
    for len in 0..=max_len {
        let total = num_tm_us.pow(len as u32) as usize;
        for mask in 0..total {
            let seq: Vec<u32> = (0..len)
                .map(|i| {
                    let digit = (mask / num_tm_us.pow(i as u32)) % num_tm_us;
                    num_nt + digit as u32
                })
                .collect();
            let local: Vec<u32> = seq.iter().map(|&x| x - num_nt).collect();
            let pda_says = m.accepts_npda(&local, 64, 1_000_000);
            let oracle_says = cfg_accepts(g, &seq);
            assert_eq!(
                pda_says, oracle_says,
                "the PDA must match the cfg_accepts oracle on global={:?}",
                seq
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "the corpus must be non-empty");
}

#[test]
fn the_sql_pda_matches_the_cfg_oracle_exhaustively() {
    let g = sql_cfg();
    let m = pushdown_rs::compile(&g).expect("compile the SQL CFG");
    assert_eq!(m.num_states, kappa(&g), "the node count is the exact kappa(G)");
    exhaustive_differential(&m, &g, 4); // the the 9-terminal alphabet, the the length 0..=4
}

#[test]
fn the_ebpf_pda_matches_the_cfg_oracle_exhaustively() {
    let g = ebpf_cfg();
    let m = pushdown_rs::compile(&g).expect("compile the eBPF CFG");
    assert_eq!(m.num_states, kappa(&g), "the node count is the exact kappa(G)");
    exhaustive_differential(&m, &g, 6); // the the 4-terminal alphabet, the the length 0..=6
}

#[test]
fn the_sql_language_matches_the_recursive_descent_oracle() {
    let g = sql_cfg();
    let m = pushdown_rs::compile(&g).expect("compile the SQL CFG");
    let cases: Vec<Vec<u32>> = vec![
        vec![SELECT, IDENT, FROM, IDENT], // the the minimal
        vec![SELECT, NUM, STAR, IDENT, FROM, IDENT], // the the *
        vec![SELECT, LPAREN, LPAREN, NUM, RPAREN, RPAREN, FROM, IDENT], // the the nested
        vec![SELECT, IDENT, FROM, IDENT, WHERE, IDENT], // the the WHERE
        vec![SELECT, FROM, IDENT], // the the invalid: the no Expr
        vec![SELECT, LPAREN, IDENT, FROM, IDENT], // the the invalid: the unbalanced
        vec![], // the the just-out: the empty
    ];
    for w in &cases {
        let pda_says = m.accepts_npda(w, 64, 100_000);
        let oracle_says = sql_oracle(w);
        assert_eq!(
            pda_says, oracle_says,
            "the SQL PDA must match the recursive-descent oracle on {:?}",
            w
        );
    }
}

#[test]
fn the_ebpf_language_matches_the_walker_oracle() {
    let g = ebpf_cfg();
    let m = pushdown_rs::compile(&g).expect("compile the eBPF CFG");
    let cases: Vec<Vec<u32>> = vec![
        vec![OP, EXIT], // the the minimal
        vec![CALL, OP, RET, EXIT], // the the one nested call
        vec![CALL, CALL, OP, RET, RET, EXIT], // the the two nested
        vec![EXIT], // the the empty Sub
        vec![CALL, EXIT], // the the invalid: the unmatched CALL
        vec![RET, EXIT], // the the invalid: the RET with the no CALL
        vec![], // the the just-out: the empty
    ];
    for w in &cases {
        let pda_says = m.accepts_npda(w, 64, 100_000);
        let oracle_says = ebpf_walker(w);
        assert_eq!(
            pda_says, oracle_says,
            "the eBPF PDA must match the walker oracle on {:?}",
            w
        );
    }
}