//! The PDA -> SVG visualization dump (the human-meaningful reference).
//!
//! Builds a few PDAs (the {a^n b^n}, the balanced-parens Dyck language, and a
//! hand-built DFA-embedded machine like the network protocols), prints a
//! console debug dump (the states, the transitions, the kappa, the determinism,
//! a trace), and writes both an SVG (the primary, zero-dep) and a DOT (the
//! graphviz)-layout) to `viz/`.
//!
//! Run: `cargo run --example viz_dump` then open `viz/*.svg` in a browser.

use pushdown_rs::compile::{Cfg, NamedCfg, kappa, rtn_state_names};
use pushdown_rs::machine::{PdaMachine, Transition};
use pushdown_rs::pda::{Dpda, Npda};
use pushdown_rs::viz;
use std::path::Path;

/// The console debug dump (the "show me the machine" reference).
fn dump_debug(m: &PdaMachine, title: &str, state_names: Option<&[String]>) {
    println!("\n=== {title} ===");
    println!(
        "  states={} inputs={} stack_syms={} transitions={} deterministic={}",
        m.num_states,
        m.num_inputs,
        m.num_stack_syms,
        m.transitions.len(),
        m.is_deterministic()
    );
    println!(
        "  start_state={} start_stack={} accepting={:?}",
        m.start_state,
        m.start_stack,
        m.accepting
    );
    println!("  transitions (q, in, top) -> (q', push):");
    for t in &m.transitions {
        let in_name = if t.a == m.num_inputs {
            "eps".to_string()
        } else {
            format!("{}", t.a)
        };
        println!(
            "    (q={}, in={}, top={}) -> (q={}, push={:?})",
            t.q, in_name, t.top, t.next_q, t.push
        );
    }
    if let Some(names) = state_names {
        println!("  state names:");
        for (i, n) in names.iter().enumerate() {
            if (i as u32) < m.num_states {
                println!("    {i} = {n}");
            }
        }
    }
}

fn main() {
    let dir = Path::new("viz");
    std::fs::create_dir_all(dir).expect("create viz dir");

    // ---- 1. The {a^n b^n} (the classic pushdown, the S -> a S b | eps) ----
    let g_anb = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2]), // S -> a S b
            (0, vec![]), // S -> eps
        ],
    );
    // The NamedCfg owns the vocabulary (the terminal_name -> a/b), so the
    // compiled machine carries m.vocab_names and the viz reads the machine's
    // own symbols (the no caller-side term array).
    let g_anb_named = NamedCfg::new(g_anb, vec!["a", "b"].iter().map(|s| s.to_string()).collect());
    let m_anb = pushdown_rs::compile(&g_anb_named).expect("compile a^n b^n");
    let anb_states = rtn_state_names(&g_anb_named);
    dump_debug(&m_anb, "the {a^n b^n} DPDA", Some(&anb_states));
    println!(
        "  kappa(G) = {} (the exact control-state count)",
        kappa(&g_anb_named)
    );
    // A trace (the decompose the closure states). Local terminals: a=0, b=1.
    println!("  trace [0, 1] (the \"ab\", the a^n b^n with n=1):");
    m_anb.trace_npda(&[0, 1], 64);
    let svg = dir.join("anbn.svg");
    let dot = dir.join("anbn.dot");
    viz::write_svg(&m_anb, Some(&anb_states), None, &svg).expect("write a^n b^n svg");
    viz::write_dot(&m_anb, Some(&anb_states), None, &dot).expect("write a^n b^n dot");
    println!("  wrote: {}  +  {}", svg.display(), dot.display());

    // ---- 2. The balanced-parens Dyck language (the S -> ( S ) S | eps) ----
    let g_dyck = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2, 0]), // S -> ( S ) S
            (0, vec![]), // S -> eps
        ],
    );
    // The NamedCfg owns the vocabulary (the terminal_name -> ( / )), so the
    // compiled machine carries m.vocab_names and the viz reads the machine's
    // own symbols (the no caller-side term array).
    let g_dyck_named = NamedCfg::new(g_dyck, vec!["(", ")"].iter().map(|s| s.to_string()).collect());
    let m_dyck = pushdown_rs::compile(&g_dyck_named).expect("compile dyck");
    let dyck_states = rtn_state_names(&g_dyck_named);
    dump_debug(&m_dyck, "the balanced-parens (Dyck) DPDA", Some(&dyck_states));
    println!(
        "  kappa(G) = {} (the exact control-state count)",
        kappa(&g_dyck_named)
    );
    let svg = dir.join("dyck.svg");
    let dot = dir.join("dyck.dot");
    viz::write_svg(&m_dyck, Some(&dyck_states), None, &svg).expect("write dyck svg");
    viz::write_dot(&m_dyck, Some(&dyck_states), None, &dot).expect("write dyck dot");
    println!("  wrote: {}  +  {}", svg.display(), dot.display());

    // ---- 3. A hand-built DFA-embedded machine (the network-protocol idiom) ----
    // The 3-state acceptor: the (a|b)* a (the ends in a). The stack is unused
    // (the single bottom symbol), so this is a DFA embedded in the PDA framework.
    let m_dfa = PdaMachine {
        num_states: 3,
        num_inputs: 2,
        num_stack_syms: 1,
        transitions: vec![
            Transition { q: 0, a: 0, top: 0, next_q: 0, push: vec![0] }, // 0 --a--> 0
            Transition { q: 0, a: 1, top: 0, next_q: 1, push: vec![0] }, // 0 --b--> 1
            Transition { q: 1, a: 0, top: 0, next_q: 2, push: vec![0] }, // 1 --a--> 2
            Transition { q: 1, a: 1, top: 0, next_q: 1, push: vec![0] }, // 1 --b--> 1
            Transition { q: 2, a: 0, top: 0, next_q: 2, push: vec![0] }, // 2 --a--> 2
            Transition { q: 2, a: 1, top: 0, next_q: 2, push: vec![0] }, // 2 --b--> 2
        ],
        accepting: vec![2],
        start_state: 0,
        start_stack: 0,
        state_provenance: None, // the hand-built (the no RTN provenance)
        vocab_names: None, // the hand-built (the no vocabulary labels)
    };
    let dfa_names = vec!["start".to_string(), "seen_b".to_string(), "accept".to_string()];
    dump_debug(&m_dfa, "the hand-built DFA-embedded PDA (the (a|b)* a)", Some(&dfa_names));
    let dfa_terms: Vec<String> = vec!["a".into(), "b".into()];
    let svg = dir.join("dfa.svg");
    let dot = dir.join("dfa.dot");
    viz::write_svg(&m_dfa, Some(&dfa_names), Some(&dfa_terms), &svg).expect("write dfa svg");
    viz::write_dot(&m_dfa, Some(&dfa_names), Some(&dfa_terms), &dot).expect("write dfa dot");
    println!("  wrote: {}  +  {}", svg.display(), dot.display());

    // A quick acceptance sanity check (the inclusive + exclusive).
    println!("\n=== acceptance sanity (the DFA) ===");
    for w in [&[0u32][..], &[1, 0], &[1, 1, 0], &[0, 0]] {
        println!(
            "  {:?} -> accepts={} (npda={})",
            w,
            m_dfa.accepts_dpda(w),
            m_dfa.accepts_npda(w, 8, 1000)
        );
    }

    println!("\nOpen the files in viz/ in a browser (the .svg) or run `dot -Tsvg` on the .dot.");
}