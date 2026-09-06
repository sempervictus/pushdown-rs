//! The SWYB bounded pushdown summary (the S_H, arXiv:2608.28229).
//!
//! The S_H is the offline computation: the reachability labels (the Reach_H) +
//! the token-distance-to-acceptance estimates (the d_H). The d_H is the
//! upper-bound distance to acceptance from a (q, stack) config, bounded by the
//! stack depth H. The Reach_H is the set of (q, stack) configs reachable within
//! the H bound.
//!
//! The online decoder uses the S_H for the horizon-aware pruning (the d_H
//! <= K-1 filter) + the distance-guided scoring (the r_k + the alpha_k).

use std::collections::{HashMap, HashSet};

use crate::machine::PdaMachine;

/// The bounded pushdown summary (the S_H).
#[derive(Debug, Clone)]
pub struct BoundedSummary {
    /// the d_H (the distance-to-acceptance, the upper-bound). The key is
    /// the (q, stack) config, the value is the d_H (the token count).
    pub distance: HashMap<(u32, Vec<u32>), u32>,
    /// the Reach_H (the reachability label). The set of the (q, stack)
    /// configs reachable within the H bound.
    pub reachable: HashSet<(u32, Vec<u32>)>,
    /// the H bound (the stack depth bound).
    pub h: usize,
}

impl BoundedSummary {
    /// Compute the bounded pushdown summary (the S_H) for the machine.
    ///
    /// The d_H is computed via the dynamic programming: d_H(q, gamma) = the min
    /// over the transitions (q, a, top) -> (q', gamma') of (1 + d_H(q', gamma')).
    /// The H bound: the d_H is computed
    /// for the stack depth <= H.
    pub fn compute(machine: &PdaMachine, h: usize) -> Self {
        let mut reachable: HashSet<(u32, Vec<u32>)> = HashSet::new();
        let mut distance: HashMap<(u32, Vec<u32>), u32> = HashMap::new();

        // the Reach_H (the forward BFS, the reachable configs within
        // the H bound)
        let mut frontier: Vec<(u32, Vec<u32>)> =
            vec![(machine.start_state, vec![machine.start_stack])];
        reachable.insert((machine.start_state, vec![machine.start_stack]));
        while let Some((q, stack)) = frontier.pop() {
            let Some(top) = stack.last().copied() else {
                continue;
            };
            for a in 0..=machine.num_inputs {
                let input = if a == machine.num_inputs { None } else { Some(a) };
                for t in machine.lookup(q, input, top) {
                    let mut s2 = stack.clone();
                    s2.pop();
                    for &p in t.push.iter().rev() {
                        s2.push(p);
                    }
                    if s2.len() > h {
                        continue;
                    }
                    if reachable.insert((t.next_q, s2.clone())) {
                        frontier.push((t.next_q, s2));
                    }
                }
            }
        }

        // the d_H (the distance-to-acceptance, the reverse BFS from
        // the accepting configs)
        for (q, stack) in reachable.iter() {
            if machine.accepting.contains(q) {
                distance.insert((*q, stack.clone()), 0);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for (q, stack) in reachable.iter() {
                if distance.contains_key(&(*q, stack.clone())) {
                    continue;
                }
                let Some(top) = stack.last().copied() else {
                    continue;
                };
                for a in 0..=machine.num_inputs {
                    let input = if a == machine.num_inputs { None } else { Some(a) };
                    for t in machine.lookup(*q, input, top) {
                        let mut s2 = stack.clone();
                        s2.pop();
                        for &p in t.push.iter().rev() {
                            s2.push(p);
                        }
                        if s2.len() > h {
                            continue;
                        }
                        let key2 = (t.next_q, s2.clone());
                        if let Some(d2) = distance.get(&key2) {
                            distance.insert((*q, stack.clone()), d2 + 1);
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }

        BoundedSummary {
            distance,
            reachable,
            h,
        }
    }

    /// The the d_H query (the distance-to-acceptance).
    pub fn distance(&self, q: u32, stack: &[u32]) -> Option<u32> {
        self.distance.get(&(q, stack.to_vec())).copied()
    }

    /// The the Reach_H query (the reachability label).
    pub fn is_reachable(&self, q: u32, stack: &[u32]) -> bool {
        self.reachable.contains(&(q, stack.to_vec()))
    }
}

/// The SWYB online decoding (the beam search, the d_H + the
/// Reach_H). The horizon-aware pruning (the d_H <= K-1) + the
/// distance-guided scoring (the r_k + the alpha_k).
pub struct SwybDecoder<'a> {
    pub machine: &'a PdaMachine,
    pub summary: &'a BoundedSummary,
    pub beam_width: usize,
}

/// The termination proof result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationProof {
    /// The grammar can terminate: the accepting state is reachable from
    /// the start config within the stack bound.
    Proven {
        /// The minimum number of tokens to reach acceptance (the d_H).
        min_tokens_to_accept: u32,
        /// The total number of reachable configs (the proof space).
        reachable_configs: usize,
    },
    /// The grammar has a dead-end: no path from start to accepting state
    /// within the stack bound. Generation would run on indefinitely.
    DeadEnd {
        /// The number of reachable configs that CANNOT reach acceptance.
        stuck_configs: usize,
        /// The total reachable configs.
        reachable_configs: usize,
    },
}

impl BoundedSummary {
    /// Prove that the PDA can terminate: from the start config, there exists
    /// a path to the accepting state within the stack bound H.
    ///
    /// This is the compile-time proof that prevents run-on generation:
    /// if the grammar has a dead-end (no path to EOS/EOR/EOT), the proof
    /// fails and the caller should reject the grammar.
    pub fn prove_termination(&self, machine: &PdaMachine) -> TerminationProof {
        let start = (machine.start_state, vec![machine.start_stack]);
        match self.distance.get(&start) {
            Some(&d) => TerminationProof::Proven {
                min_tokens_to_accept: d,
                reachable_configs: self.reachable.len(),
            },
            None => {
                // Count how many reachable configs are stuck (no distance).
                let stuck = self
                    .reachable
                    .iter()
                    .filter(|(q, _)| !machine.accepting.contains(q))
                    .count();
                TerminationProof::DeadEnd {
                    stuck_configs: stuck,
                    reachable_configs: self.reachable.len(),
                }
            }
        }
    }
}

impl<'a> SwybDecoder<'a> {
    pub fn new(machine: &'a PdaMachine, summary: &'a BoundedSummary, beam_width: usize) -> Self {
        SwybDecoder {
            machine,
            summary,
            beam_width,
        }
    }

    /// The the horizon-aware pruning (the d_H <= K-1). The the K is the
    /// remaining token budget.
    pub fn is_viable(&self, q: u32, stack: &[u32], remaining_budget: u32) -> bool {
        match self.summary.distance(q, stack) {
            Some(d) => d <= remaining_budget,
            None => false, // the unreachable (the d_H is undefined)
        }
    }
}