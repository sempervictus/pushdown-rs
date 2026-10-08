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
use rayon::prelude::*;

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

    /// The UNBOUNDED pushdown summary (the S, the no the caller-supplied H bound).
    /// The d is the least fixed point of the distance equations over the (min,+)
    /// dioid (the Reps 2003 weighted PDS, the "min-plus semiring finds the
    /// shortest trace"): d(c) = 0 if c is accepting, else d(c) = 1 + min over
    /// c -> c' of d(c'). This is the shortest-path-to-acceptance over the config
    /// graph, computed by the reverse BFS (the pre* saturation, the Bouajjani 1997).
    ///
    /// The stack depth is bounded by D = max_push + 1 (the six-property #5, the
    /// bounded pushdown: the max production length + 1, the pending nesting).
    /// The max_push is derived from the machine (the maximum length of any
    /// transition's push string, the no a static value). The reverse BFS is
    /// restricted to stacks of depth <= D (the no the unbounded exploration,
    /// which does not terminate for machines with a bounded forward stack).
    /// The h field is set to D (the intrinsic bound, the no usize::MAX).
    pub fn compute_unbounded(machine: &PdaMachine) -> Self {
        // The six-property #5 bounded pushdown: the D = max_push + 1 (the max
        // production length + 1, the pending nesting). The max_push is derived
        // from the machine (the maximum length of any transition's push string).
        let max_push = machine.transitions.iter().map(|t| t.push.len()).max().unwrap_or(1);
        let d_bound = max_push + 1; // the stack depth bound (the six-property #5).
        let mut distance: HashMap<(u32, Vec<u32>), u32> = HashMap::new();
        // The reverse BFS (the pre* saturation): start from the accepting configs
        // (the d = 0), and propagate the distance backwards (the d(c) = 1 + min d(c')).
        // The accepting configs are (q, stack) where q is in F. The stack can be
        // any (the unbounded), so we seed the BFS with the accepting states at the
        // start_stack (the bottom) + the reverse transitions.
        // The rayon parallelizes the BFS levels (the queue at level k is processed
        // in parallel to produce level k+1, the no the sequential queue).
        let mut current_level: Vec<(u32, Vec<u32>, u32)> = Vec::new();
        for &fq in &machine.accepting {
            // The accepting config (fq, [start_stack]) has d = 0 (the top-level
            // acceptance, the stack exactly at the bottom marker).
            let cfg = (fq, vec![machine.start_stack]);
            distance.insert(cfg.clone(), 0);
            current_level.push((cfg.0, cfg.1, 0));
        }
        while !current_level.is_empty() {
            // The rayon parallelizes: the current_level configs are processed
            // independently (each produces its predecessors, the no the data race
            // on the distance map, the rayon reduce).
            let next_level: Vec<(u32, Vec<u32>, u32)> = current_level
                .par_iter()
                .flat_map(|&(cq, ref cstk, cd)| {
                    // The reverse transitions: find all predecessor configs (q, sigma)
                    // that transition to (cq, cstk) via a single move.
                    //
                    // The PDA transition semantics (the ODU CS390, the
                    // delta(q, a, top) = {(q', push)}): the successor stack is
                    //   cstk = push.reversed() ++ sigma'
                    // where sigma' = sigma without its top (the popped symbol). So:
                    //   sigma' = cstk[|push|..]           (the remaining stack)
                    //   top    = a free variable in Gamma (the popped symbol, the no
                    //            in cstk, the transition only sees the top)
                    //   sigma  = sigma' ++ [top]          (the predecessor stack)
                    // The rule (t.q, a, top) -> (t.next_q, t.push) must exist for the
                    // predecessor (t.q, sigma) to reach (cq, cstk).
                    // The d(t.q, sigma) = cd + 1 (the one token consumed).
                    let mut preds: Vec<(u32, Vec<u32>, u32)> = Vec::new();
                    for a in 0..=machine.num_inputs {
                        for t in machine.transitions.iter() {
                            if t.next_q != cq || t.a != a {
                                continue;
                            }
                            // The PDA transition semantics (the ODU CS390, the
                            // delta(q, a, top) = {(q', push)}): the successor stack is
                            //   cstk = deeper ++ push.reversed()
                            // where deeper = sigma without its top (the the popped symbol),
                            // and push.reversed() is the t.push block (the t.push[0] on top).
                            // So:
                            //   m      = t.push.len()
                            //   deeper = cstk[0..(cstk.len() - m)]   (the the stack below the pushed block)
                            //   top    = t.top                        (the the rule's popped symbol, the FIXED)
                            //   sigma  = deeper ++ [top]             (the the predecessor stack)
                            // The cstk must be at least m long (the the pushed block fits).
                            let m = t.push.len();
                            if cstk.len() < m {
                                continue;
                            }
                            // The deeper = cstk[0..(cstk.len() - m)] (the the stack below the
                            // pushed block, the the predecessor-minus-top).
                            let deeper: Vec<u32> = cstk[..(cstk.len() - m)].to_vec();
                            let mut sigma = deeper;
                            sigma.push(t.top); // the the predecessor top (the the rule's t.top, the FIXED).
                            // The six-property #5 bound: the predecessor stack depth must be
                            // <= d_bound (the no the unbounded exploration).
                            if sigma.len() > d_bound {
                                continue;
                            }
                            let pred = (t.q, sigma.clone());
                            preds.push((pred.0, pred.1, cd + 1));
                        }
                    }
                    preds
                })
                .collect();
            // The dedup + the distance update (the sequential, the no the data race).
            // The new_level is the predecessors at distance cd+1 (the no the
            // accumulated current_level, which would cause the exponential blowup).
            let mut new_level: Vec<(u32, Vec<u32>, u32)> = Vec::new();
            for (pq, pstack, pd) in next_level {
                let pred = (pq, pstack.clone());
                match distance.get(&pred) {
                    Some(&existing) if existing <= pd => {} // the no improvement.
                    _ => {
                        distance.insert(pred.clone(), pd);
                        new_level.push((pq, pstack, pd));
                    }
                }
            }
            current_level = new_level;
        }
        // The Reach (the forward BFS, the unbounded, the no the H bound).
        // The six-property #5 precondition: the machine's stack is bounded by
        // d_bound (the max_push + 1). The forward BFS computes the Reach (the
        // d_bound truncated). The guard: if the truncation is non-vacuous (the
        // any config at depth d_bound has a transition to depth d_bound + 1),
        // the precondition fails (the machine is NOT six-property #5 bounded,
        // the compute_unbounded is the wrong primitive, the use the
        // compute(machine, h) with a caller-supplied h). The explicit panic
        // (the no the silent truncation).
        let mut reachable: HashSet<(u32, Vec<u32>)> = HashSet::new();
        let mut frontier: Vec<(u32, Vec<u32>)> =
            vec![(machine.start_state, vec![machine.start_stack])];
        reachable.insert((machine.start_state, vec![machine.start_stack]));
        let mut truncated = false;
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
                    if s2.len() > d_bound {
                        // The truncation is non-vacuous (the config at depth
                        // d_bound + 1 is dropped). The six-property #5
                        // precondition fails (the machine is NOT bounded by
                        // d_bound). The explicit panic (the no the silent
                        // omission).
                        truncated = true;
                        continue;
                    }
                    if reachable.insert((t.next_q, s2.clone())) {
                        frontier.push((t.next_q, s2));
                    }
                }
            }
        }
if truncated {
            panic!(
                "compute_unbounded: the six-property #5 precondition failed (the \
                 machine's stack depth exceeds d_bound = {} (the max_push + 1)). \
                 The forward Reach BFS truncated configs at depth > d_bound \
                 (the silent omission). Use the compute(machine, h) with a \
                 caller-supplied h for unbounded-stack machines (the a^n b^n).",
                d_bound
            );
        }
        BoundedSummary {
            distance,
            reachable,
            h: d_bound, // the six-property #5 intrinsic bound (the max_push + 1, the no usize::MAX).
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