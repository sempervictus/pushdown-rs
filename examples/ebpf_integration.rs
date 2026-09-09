//! The eBPF integration: the eBPF program CFG + the call-depth walker oracle +
//! the PDA + the differential + the viz dump.
//!
//! The eBPF framing (the well-nested call/return subset, the genuine pushdown):
//! a program is a sequence of instructions where the CALL/RET pairs are
//! well-nested (the PDA stack tracks the call depth), terminated by a single
//! EXIT. This is a DCFL (the balanced-parens shape), so it exercises the
//! pushdown (a DFA cannot do it).
//!
//!   Prog -> Sub EXIT
//!   Sub  -> CALL Sub RET | OP Sub
//!
//! FALSIFIER (the stated gap): real eBPF calls are label-based and NOT strictly
//! nested. This models the well-nested subset only.

use pushdown_rs::compile::{Cfg, kappa, rtn_state_names};
use pushdown_rs::pda::{Dpda, Npda};
use pushdown_rs::viz;

// The local terminal ids (the instruction classes).
const CALL: u32 = 0;
const RET: u32 = 1;
const OP: u32 = 2;
const EXIT: u32 = 3;

const TERM_NAMES: &[&str] = &["CALL", "RET", "OP", "EXIT"];

/// The eBPF program CFG (the well-nested call/return, the DCFL).
/// Global symbol ids: the nonterminals 0..3 (the Prog=0, the Sub=1, the Unit=2),
/// the terminals 3..7 (the CALL=3, the RET=4, the OP=5, the EXIT=6).
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

/// The call-depth walker oracle (the independent register, the zero shared code
/// with the PDA). A valid program: the last token is EXIT, and the prefix is a
/// balanced CALL/RET sequence with OPs anywhere (the prefix may be empty, the
/// Sub -> eps).
fn ebpf_walker(tokens: &[u32]) -> bool {
    match tokens.last() {
        Some(&EXIT) => is_sub(&tokens[..tokens.len() - 1]),
        _ => false,
    }
}

fn is_sub(prefix: &[u32]) -> bool {
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
            _ => return false, // no EXIT in the prefix
        }
    }
    depth == 0
}

fn show(toks: &[u32]) -> String {
    toks.iter()
        .map(|t| TERM_NAMES.get(*t as usize).copied().unwrap_or("?"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    println!("=== The eBPF integration: the well-nested call/return CFG + the walker + the PDA ===\n");

    let g = ebpf_cfg();
    let m = pushdown_rs::compile(&g).expect("compile the eBPF CFG");
    println!(
        "the PDA: {} states (kappa={}), {} transitions, deterministic={}",
        m.num_states,
        kappa(&g),
        m.transitions.len(),
        m.is_deterministic()
    );

    // The differential corpus (the valid + the invalid + the boundary).
    let corpus: Vec<Vec<u32>> = vec![
        vec![OP, EXIT], // the minimal (the one OP + the EXIT)
        vec![CALL, OP, RET, EXIT], // the one nested call
        vec![CALL, CALL, OP, RET, RET, EXIT], // the two nested calls
        vec![OP, OP, OP, EXIT], // the many OPs
        vec![CALL, RET, EXIT], // the empty call body
        vec![EXIT], // the valid: the empty Sub (the Sub -> eps) + the EXIT
        vec![CALL, EXIT], // the invalid: the unmatched CALL
        vec![RET, EXIT], // the invalid: the RET with the no CALL
        vec![OP, CALL, RET], // the invalid: the no EXIT
        vec![], // the empty (the just-out boundary)
    ];

    println!("\n=== The differential: the PDA == the call-depth walker ===");
    let mut agree = 0;
    for w in &corpus {
        let pda_says = m.accepts_npda(w, 64, 100_000);
        let oracle_says = ebpf_walker(w);
        let ok = pda_says == oracle_says;
        if ok {
            agree += 1;
        }
        println!(
            "  {:30} -> pda={} walker={} {}",
            show(w),
            pda_says,
            oracle_says,
            if ok { "OK" } else { "MISMATCH" }
        );
    }
    println!("\n  {}/{} agree (the the 100% gate)", agree, corpus.len());

    // The viz dump (the the human-meaningful SVG reference).
    let states = rtn_state_names(&g);
    let terms: Vec<String> = TERM_NAMES.iter().map(|s| s.to_string()).collect();
    let dir = std::path::Path::new("viz");
    std::fs::create_dir_all(dir).ok();
    let svg = dir.join("ebpf_program.svg");
    let dot = dir.join("ebpf_program.dot");
    viz::write_svg(&m, Some(&states), Some(&terms), &svg).expect("write ebpf svg");
    viz::write_dot(&m, Some(&states), Some(&terms), &dot).expect("write ebpf dot");
    println!("\n=== The viz dump ===");
    println!("  wrote: {}  +  {}", svg.display(), dot.display());
}