use pushdown_rs::compile::Cfg;
use pushdown_rs::pda::Dpda;
use pushdown_rs::Npda;

use pushdown_rs::compile::{NamedCfg, rtn_state_names};
use pushdown_rs::machine::PdaMachine;
use pushdown_rs::viz;

/// Write the viz dump (the SVG + the DOT) for a compiled PDA. The terminal
/// labels come from the machine's own vocab_names (the NamedCfg), so we pass
/// None for the caller terms (the single source of truth).
fn dump_viz(m: &PdaMachine, states: &[String], name: &str) {
    let dir = std::path::Path::new("viz");
    std::fs::create_dir_all(dir).ok();
    let svg = dir.join(format!("{name}.svg"));
    let dot = dir.join(format!("{name}.dot"));
    viz::write_svg(m, Some(states), None, &svg).expect("write svg");
    viz::write_dot(m, Some(states), None, &dot).expect("write dot");
    println!("\n=== The viz dump ({name}) ===");
    println!("  wrote: {}  +  {}", svg.display(), dot.display());
}
fn main() {
    // the S -> a S b | eps (the {a^n b^n})
    let g = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0, 2]), // S -> a S b (the a=1, the S=0, the b=2)
            (0, vec![]), // S -> eps
        ],
    );
    let g_named = NamedCfg::new(g, vec!["a", "b"].iter().map(|s| s.to_string()).collect());
    let m = pushdown_rs::compile(&g_named).expect("compile");
    println!("num_states={} num_inputs={} num_stack_syms={}", m.num_states, m.num_inputs, m.num_stack_syms);
    println!("accepting={:?} start_state={} start_stack={}", m.accepting, m.start_state, m.start_stack);
    for t in &m.transitions {
        println!("  ({}, in {}, top {}) -> ({}, push {:?})", t.q, t.a, t.top, t.next_q, t.push);
    }
    println!("accepts [1,2] (ab) = {}", m.accepts_npda(&[1, 2], 64, 100_000));
    println!("accepts [1,1,2,2] (aabb) = {}", m.accepts_npda(&[1, 1, 2, 2], 64, 100_000));
    println!("\n=== TRACE [1,2] (the decompose the closure states) ===");
    m.trace_npda(&[1, 2], 64);
    dump_viz(&m, &rtn_state_names(&g_named), "anbn");

    // the [a-z]+ regex case: the empty-input mismatch exposure
    println!("\n=== the regex [a-z]+ empty-input exposure ===");
    // the [a-z]+ = one-or-more. The CFG: S -> a S | a  (the no epsilon, the
    // the one-or-more). If the compile added an S -> eps, the empty is wrongly accepted.
    let rg = Cfg::new(
        1,
        2,
        0,
        vec![
            (0, vec![1, 0]), // S -> a S (the one-or-more, the recursive)
            (0, vec![1]), // S -> a (the base)
        ],
    );
    println!("  productions: {:?}", rg.productions);
    let rg_named = NamedCfg::new(rg, vec!["a", "b"].iter().map(|s| s.to_string()).collect());
    let rm = pushdown_rs::compile(&rg_named).expect("compile");
    println!("  accepts_dpda([])  = {}", rm.accepts_dpda(&[]));
    println!("  accepts_npda([])  = {}", rm.accepts_npda(&[], 64, 100_000));
    println!("  oracle cfg_accepts([]) = {}", pushdown_rs::oracle::cfg_accepts(&rg_named.cfg, &[]));
    println!("  accepts_dpda([1]) = {}", rm.accepts_dpda(&[1]));
    println!("  oracle cfg_accepts([1]) = {}", pushdown_rs::oracle::cfg_accepts(&rg_named.cfg, &[1]));
    dump_viz(&rm, &rtn_state_names(&rg_named), "regex_one_or_more");
    println!("\n=== the regex [a-z]+ PDA (the real llguidance CFG) ===");
    // the regex via: the start -> start#2 (the [a-z]+ terminal)
    let rg2 = Cfg::new(
        2, // the N = {start, start#2}
        2, // the Sigma = {the [a-z],, the ?}
        0, // the S = start
        vec![
            (0, vec![2]), // start -> start#2 (the terminal 2)
        ],
    );
    let rg2_named = NamedCfg::new(rg2, vec!["[a-z]", "?"].iter().map(|s| s.to_string()).collect());
    let rm2 = pushdown_rs::compile(&rg2_named).expect("compile");
    println!("  states={} inputs={} transitions={}", rm2.num_states, rm2.num_inputs, rm2.transitions.len());
    for t in &rm2.transitions {
        println!("    ({}, in {}, top {}) -> ({}, push {:?})", t.q, t.a, t.top, t.next_q, t.push);
    }
    println!("  accepts([]) = {}", rm2.accepts(&[]));
    println!("  accepts([2]) = {}", rm2.accepts(&[2]));
    rm2.trace_npda(&[], 64);
    dump_viz(&rm2, &rtn_state_names(&rg2_named), "regex_llguidance");
}