//! An INDEPENDENT CFG language-membership oracle.
//!
//! This is the ground truth the PDA is verified against. It is structurally
//! independent of the RTN compilation (zero shared code): it is the naive
//! recursive definition of "does nonterminal A derive the input substring",
//! which is the definition definition of a context-free grammar. If the PDA
//! (compiled from the CFG) and this oracle disagree on any input, the PDA is
//! wrong. This kills the circularity of comparing the PDA to itself.

use std::collections::HashMap;

use crate::compile::Cfg;

/// Does the CFG `g` derive the terminal string `input` from its start symbol?
/// The naive recursive definition (the O(n^4) worst case, fine for test
/// grammars). Fully independent of any PDA construction.
pub fn cfg_accepts(g: &Cfg, input: &[u32]) -> bool {
    let mut memo: HashMap<(u32, usize, usize), bool> = HashMap::new();
    derive(g, input, &mut memo, g.start, 0, input.len())
}

/// Can nonterminal `nt` derive input[i..j]?
fn derive(g: &Cfg, input: &[u32], memo: &mut HashMap<(u32, usize, usize), bool>, nt: u32, i: usize, j: usize) -> bool {
    if let Some(&v) = memo.get(&(nt, i, j)) {
        return v;
    }
    let mut result = false;
    for (lhs, rhs) in &g.productions {
        if *lhs != nt {
            continue;
        }
        if rhs.is_empty() {
            // the epsilon production: derives the empty string only
            if i == j {
                result = true;
                break;
            }
            continue;
        }
        if split(g, input, memo, rhs, i, j) {
            result = true;
            break;
        }
    }
    memo.insert((nt, i, j), result);
    result
}

/// Can the symbol sequence `rhs` derive input[i..j]? (the split recursion)
fn split(g: &Cfg, input: &[u32], memo: &mut HashMap<(u32, usize, usize), bool>, rhs: &[u32], i: usize, j: usize) -> bool {
    if rhs.is_empty() {
        return i == j;
    }
    let (first, rest) = (&rhs[0], &rhs[1..]);
    for k in i..=j {
        if symbol_derives(g, input, memo, first, i, k) && split(g, input, memo, rest, k, j) {
            return true;
        }
    }
    false
}

/// Does the single symbol `s` (terminal or nonterminal) derive input[i..j]?
fn symbol_derives(g: &Cfg, input: &[u32], memo: &mut HashMap<(u32, usize, usize), bool>, s: &u32, i: usize, j: usize) -> bool {
    if *s < g.num_nonterminals {
        derive(g, input, memo, *s, i, j)
    } else {
        // a terminal: it derives exactly the one-symbol substring
        j - i == 1 && input[i] == *s
    }
}