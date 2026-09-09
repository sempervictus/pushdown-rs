//! The SQL integration: the SQL-subset CFG (the our code) + the recursive-descent
//! parser (the oracle, the independent register) + the PDA (the RTN compilation)
//! + the differential (the PDA == the parser) + the viz dump.
//!
//! This proves the generic function: the PDA crate is generic over the grammar
//! source (the CFG), and the recursive-descent parser is the oracle (the ground
//! truth, the zero shared code with the PDA).
//!
//! The SQL subset (the left-factored, the non-left-recursive, the deterministic):
//!   Stmt     -> Select
//!   Select   -> SELECT Expr FROM IDENT WhereOpt
//!   WhereOpt -> WHERE Expr | eps
//!   Expr     -> Term ExprTail
//!   ExprTail -> OR Term ExprTail | eps
//!   Term     -> Factor TermTail
//!   TermTail -> * Factor TermTail | eps
//!   Factor   -> ( Expr ) | IDENT | NUM
//!
//! The nesting (the Factor -> ( Expr )) exercises the PDA stack (the DCFL).
//!
//! Run: `cargo run --example sql_integration`.

use pushdown_rs::compile::{Cfg, kappa, rtn_state_names};
use pushdown_rs::pda::{Dpda, Npda};
use pushdown_rs::viz;

// The local terminal ids (the the 0..num_tm).
const SELECT: u32 = 0;
const FROM: u32 = 1;
const WHERE: u32 = 2;
const OR: u32 = 3;
const STAR: u32 = 4;
const LPAREN: u32 = 5;
const RPAREN: u32 = 6;
const IDENT: u32 = 7;
const NUM: u32 = 8;

const TERM_NAMES: &[&str] = &[
    "SELECT", "FROM", "WHERE", "OR", "*", "(", ")", "IDENT", "NUM",
];

/// The SQL-subset CFG (the the global symbol ids: the nonterminals 0..8, the
/// terminals 8..17).
fn sql_cfg() -> Cfg {
    Cfg::new(
        8, // the N = {Stmt, Select, WhereOpt, Expr, ExprTail, Term, TermTail, Factor}
        9, // the Sigma = {SELECT, FROM, WHERE, OR, *, (, ), IDENT, NUM}
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

/// The recursive-descent oracle (the the independent register, the the zero
/// shared code with the PDA). Parses the local terminal sequence.
struct Parser<'a> {
    toks: &'a [u32],
    pos: usize,
}

impl<'a> Parser<'a> {
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
            true // the eps
        }
    }
    fn parse_select(&mut self) -> bool {
        self.expect(SELECT)
            && self.parse_expr()
            && self.expect(FROM)
            && self.expect(IDENT)
            && self.parse_whereopt()
    }
    /// The whole-statement parse (the the accept iff the the full input is consumed).
    fn accepts(&mut self) -> bool {
        let ok = self.parse_select();
        ok && self.pos == self.toks.len()
    }
}

fn sql_oracle(toks: &[u32]) -> bool {
    Parser { toks, pos: 0 }.accepts()
}

/// A human-readable token sequence (the the local ids -> the names).
fn show(toks: &[u32]) -> String {
    toks.iter()
        .map(|t| TERM_NAMES.get(*t as usize).copied().unwrap_or("?"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    println!("=== The SQL integration: the SQL-subset CFG + the recursive-descent oracle + the PDA ===\n");

    let g = sql_cfg();
    let m = pushdown_rs::compile(&g).expect("compile the SQL CFG");
    println!(
        "the PDA: {} states (kappa={}), {} transitions, deterministic={}",
        m.num_states,
        kappa(&g),
        m.transitions.len(),
        m.is_deterministic()
    );

    // The differential corpus (the the valid + the invalid + the boundary).
    let corpus: Vec<Vec<u32>> = vec![
        vec![SELECT, IDENT, FROM, IDENT], // the SELECT x FROM t (the the minimal)
        vec![SELECT, NUM, FROM, IDENT], // the SELECT 1 FROM t
        vec![SELECT, NUM, STAR, IDENT, FROM, IDENT], // the SELECT 1*x FROM t
        vec![SELECT, LPAREN, IDENT, RPAREN, FROM, IDENT], // the SELECT (x) FROM t
        vec![SELECT, LPAREN, LPAREN, NUM, RPAREN, RPAREN, FROM, IDENT], // the nested parens
        vec![SELECT, IDENT, FROM, IDENT, WHERE, IDENT], // the SELECT x FROM t WHERE y
        vec![SELECT, IDENT, OR, IDENT, FROM, IDENT], // the SELECT x OR y FROM t
        vec![SELECT, FROM, IDENT], // the invalid: the no Expr
        vec![SELECT, IDENT, FROM], // the invalid: the no WHERE operand / the truncated
        vec![SELECT, LPAREN, IDENT, FROM, IDENT], // the invalid: the unbalanced paren
        vec![SELECT, IDENT, FROM, IDENT, WHERE], // the invalid: the WHERE with the no Expr
        vec![], // the empty (the the just-out boundary)
    ];

    println!("\n=== The differential: the PDA == the recursive-descent oracle ===");
    let mut agree = 0;
    for w in &corpus {
        let pda_says = m.accepts_npda(w, 64, 100_000);
        let oracle_says = sql_oracle(w);
        let ok = pda_says == oracle_says;
        if ok {
            agree += 1;
        }
        println!(
            "  {:40} -> pda={} oracle={} {}",
            show(w),
            pda_says,
            oracle_says,
            if ok { "OK" } else { "MISMATCH" }
        );
    }
    println!("\n  {}/{} agree (the the 100% gate)", agree, corpus.len());

    // The viz dump (the the human-meaningful SVG reference, the the Task-1 viz).
    let states = rtn_state_names(&g);
    let terms: Vec<String> = TERM_NAMES.iter().map(|s| s.to_string()).collect();
    let dir = std::path::Path::new("viz");
    std::fs::create_dir_all(dir).ok();
    let svg = dir.join("sql_subset.svg");
    let dot = dir.join("sql_subset.dot");
    viz::write_svg(&m, Some(&states), Some(&terms), &svg).expect("write sql svg");
    viz::write_dot(&m, Some(&states), Some(&terms), &dot).expect("write sql dot");
    println!("\n=== The viz dump ===");
    println!("  wrote: {}  +  {}", svg.display(), dot.display());
}