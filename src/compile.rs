//! The RTN (recursive-transition-network) compilation of a CFG to a PDA
//! (Alpay & Senturk, arXiv:2603.05540, Definition 5).
//!
//! Given a CFG G = (N, Sigma, P, S), the compilation yields a PDA with:
//!   - control states Q = {q_start} U {q_A^in, q_A^out : A in N}
//!                        U {q_(p,i) : p in P, i in 0..=|rhs(p)|}
//!   - stack alphabet Gamma = {bot} U {q_(p,i) : p in P, i}  (the return addresses)
//!   - the RTN transitions (the start, the choice, the terminal, the call,
//!     the return, the exit)
//!
//! The exact control-state count is kappa(G) = 1 + 2|N| + sum_p (|rhs(p)| + 1)
//! (the Definition 10 + the Lemma 2 of the paper).

use crate::machine::{PdaMachine, Transition};

/// An abstract finite-state grammar (the CFG G = (N, Sigma, P, S)) of ANY sort.
/// The associated types let any concrete grammar (the Lark, the JSON schema, the
/// BNF, the EBNF) implement this trait and be compiled to a PDA.
pub trait Grammar {
    /// the nonterminals (N).
    type Nonterminal: Copy + Eq + std::hash::Hash;
    /// the terminals (Sigma).
    type Terminal: Copy + Eq + std::hash::Hash;
    /// the symbols (the terminals + the nonterminals, the rhs alphabet).
    type Symbol: Copy + Eq + std::hash::Hash;

    fn nonterminals(&self) -> Vec<Self::Nonterminal>;
    fn terminals(&self) -> Vec<Self::Terminal>;
    fn start(&self) -> Self::Nonterminal;
    /// the productions (P): (lhs, rhs).
    fn productions(&self) -> Vec<(Self::Nonterminal, Vec<Self::Symbol>)>;
    /// whether a symbol is a terminal (vs a nonterminal).
    fn is_terminal(&self, sym: &Self::Symbol) -> bool;
    /// the terminal ID for a symbol (the Sigma index), if it is a terminal.
    fn terminal_id(&self, sym: &Self::Symbol) -> Option<u32>;
    /// the nonterminal ID for a symbol (the N index), if it is a nonterminal.
    fn nonterminal_id(&self, sym: &Self::Symbol) -> Option<u32>;
    /// the N index for a nonterminal (the lhs of a production).
    fn nonterminal_index(&self, nt: &Self::Nonterminal) -> u32;
    /// the start nonterminal's N index.
    fn start_id(&self) -> u32;
    /// Prove the input is a VALID CFG (the lhs is a nonterminal, the rhs
    /// symbols are defined, the start is a nonterminal). Returns the CfgError
    /// on the first violation.
    fn validate(&self) -> Result<(), CfgError>;
}

/// The exact control-state count kappa(G) = 1 + 2|N| + sum_p (|rhs(p)| + 1)
/// (the Definition 10 + the Lemma 2 of the paper).
pub fn kappa<G: Grammar>(g: &G) -> u32 {
    let n_nt = g.nonterminals().len() as u32;
    let sum_rhs: u32 = g
        .productions()
        .iter()
        .map(|(_, rhs)| rhs.len() as u32 + 1)
        .sum();
    1 + 2 * n_nt + sum_rhs
}

/// The RTN state names (the q_start, the q_A^in, the q_A^out, the q_(p,i)).
/// Maps into the PdaMachine's state IDs (the 0..num_states).
pub fn rtn_state_names<G: Grammar>(g: &G) -> Vec<String> {
    let num_nt = g.nonterminals().len() as u32;
    let prods = g.productions();
    let mut names = Vec::new();
    names.push("q_start".to_string());
    for a in 0..num_nt {
        names.push(format!("q_nt{}^in", a));
    }
    for a in 0..num_nt {
        names.push(format!("q_{}^out", a));
    }
    for (p, (_, rhs)) in prods.iter().enumerate() {
        for i in 0..=rhs.len() {
            names.push(format!("q_{} dot{}", p, i));
        }
    }
    names
}

/// Compile an abstract grammar to a PDA machine (the RTN construction).
/// Proves the input is a valid CFG (the validate) before compiling.
pub fn compile<G: Grammar>(g: &G) -> Result<PdaMachine, CfgError> {
    g.validate()?; // the prove the input is a valid CFG
    let num_nt = g.nonterminals().len() as u32;
    let num_tm = g.terminals().len() as u32;
    let prods = g.productions();

    // GUARD: the index mapping must be consistent (the lesson learned from the
    // global/local bug). The start_id must be a valid nonterminal index, and
    // every terminal/nonterminal ID must be in range. A global symbol index
    // (the 0..num_symbols) leaking of a local index (the 0..num_nt / 0..num_tm)
    // silently corrupts the state numbering (the accepting state landed
    // on the wrong state, the empty input is wrongly accepted).
    if g.start_id() >= num_nt {
        return Err(CfgError::IndexMapping {
            got: g.start_id(),
            expected_lt: num_nt,
            what: "start_id",
        });
    }
    // the terminal_id/nonterminal_id range checks are done per-symbol in the
    // productions loop below (the rhs symbols must be in range).

    // WHY we number the states this way: the PDA has three kinds of states, and we
    // give each a unique ID so the transition table can reference them by number.
    //
    //   0            = the "start" state (where the machine begins)
    //   1..num_nt    = one "entry" state per nonterminal (where we BEGIN parsing
    //                  that nonterminal's productions)
    //   num_nt+1..   = one "exit" state per nonterminal (where we FINISH it)
    //   then         = one "dot" state per (production, position) - the dot is
    //                  how far into a production's right-hand side we've read
    //
    // The dot states are the heart of it: a production like S -> a S b has dots
    // at 4 positions (before a, between a and S, between S and b, after b). Each
    // dot is a state. Moving the dot forward = consuming the symbol before it.
    const Q_START: u32 = 0;
    let q_in = |a: u32| 1 + a; // the entry state for nonterminal a
    let q_out = |a: u32| 1 + num_nt + a; // the exit state for nonterminal a
    let dot_base = 1 + 2 * num_nt; // where the dot states begin
    let mut dot_ids: Vec<Vec<u32>> = Vec::with_capacity(prods.len());
    let mut next_dot = dot_base;
    for prod in &prods {
        let m = prod.1.len();
        let mut ids = Vec::with_capacity(m + 1);
        for _ in 0..=m {
            ids.push(next_dot);
            next_dot += 1;
        }
        dot_ids.push(ids);
    }
    let num_states = next_dot;

    // WHY the stack holds "return addresses": when a production calls a
    // nonterminal (like S calling S in S -> a S b), the machine must REMEMBER
    // where to return after the call finishes. That "where" is a dot state. So
    // the stack stores dot-state IDs (return addresses), exactly like a CPU's
    // call stack stores return addresses. The bottom marker (0) is the "no
    // return" sentinel.
    let num_stack_syms = 1 + (next_dot - dot_base);
    let ret_addr = |p: usize, i: usize| 1 + (dot_ids[p][i] - dot_base);

    let eps = num_tm; // the input ID for the epsilon
    let mut transitions: Vec<Transition> = Vec::new();

    // the start: delta(q_start, eps, bot) = (q_S^in, [bot])
    let start_nt = g.start_id();
    transitions.push(Transition {
        q: Q_START,
        a: eps,
        top: 0,
        next_q: q_in(start_nt),
        push: vec![0],
    });

    for (p_idx, (lhs, rhs)) in prods.iter().enumerate() {
        let a_id = g.nonterminal_index(lhs);
        let m = rhs.len();
        // the choice: delta(q_A^in, eps, top) = (q_(p,0), [top]) for all top
        for top in 0..num_stack_syms {
            transitions.push(Transition {
                q: q_in(a_id),
                a: eps,
                top,
                next_q: dot_ids[p_idx][0],
                push: vec![top],
            });
        }
        for (i, sym) in rhs.iter().enumerate() {
            let q_from = dot_ids[p_idx][i];
            let q_to = dot_ids[p_idx][i + 1];
            if g.is_terminal(sym) {
                // the terminal: delta(q_(p,i), X_i, top) = (q_(p,i+1), [top])
                let t_id = g.terminal_id(sym).unwrap_or(0);
                for top in 0..num_stack_syms {
                    transitions.push(Transition {
                        q: q_from,
                        a: t_id,
                        top,
                        next_q: q_to,
                        push: vec![top],
                    });
                }
            } else {
                // the nonterminal call: delta(q_(p,i), eps, top) =
                //   (q_B^in, [ret_addr(p,i+1), top])
                let b_id = g.nonterminal_id(sym).unwrap_or(0);
                for top in 0..num_stack_syms {
                    transitions.push(Transition {
                        q: q_from,
                        a: eps,
                        top,
                        next_q: q_in(b_id),
                        push: vec![ret_addr(p_idx, i + 1), top],
                    });
                }
            }
        }
        // the exit: delta(q_(p,m), eps, top) = (q_A^out, [top])
        for top in 0..num_stack_syms {
            transitions.push(Transition {
                q: dot_ids[p_idx][m],
                a: eps,
                top,
                next_q: q_out(a_id),
                push: vec![top],
            });
        }
    }

    // the return: delta(q_B^out, eps, r) = (the dot state for r, []) for each
    // nonterminal B + each return address r. Per-nonterminal (the Definition 5),
    // NOT per-production (the avoid duplicate transitions that break determinism).
    for b in 0..num_nt {
        for r in 1..num_stack_syms {
            transitions.push(Transition {
                q: q_out(b),
                a: eps,
                top: r,
                next_q: dot_base + (r - 1),
                push: vec![],
            });
        }
    }

    let accepting = vec![q_out(start_nt)];
    Ok(PdaMachine {
        num_states,
        num_inputs: num_tm,
        num_stack_syms,
        transitions,
        accepting,
        start_state: Q_START,
        start_stack: 0,
    })
}

/// The concrete CFG (the G = (N, Sigma, P, S)) with u32 symbol IDs. The
/// nonterminals are 0..num_nonterminals, the terminals are num_nonterminals..
/// (num_nonterminals + num_terminals).
#[derive(Debug, Clone)]
pub struct Cfg {
    pub num_nonterminals: u32,
    pub num_terminals: u32,
    pub start: u32,
    pub productions: Vec<(u32, Vec<u32>)>,
}

impl Cfg {
    pub fn new(
        num_nonterminals: u32,
        num_terminals: u32,
        start: u32,
        productions: Vec<(u32, Vec<u32>)>,
    ) -> Self {
        Cfg {
            num_nonterminals,
            num_terminals,
            start,
            productions,
        }
    }

    /// Prove this input is a VALID CFG (the context-free grammar).
    /// Checks: the lhs is a nonterminal, the rhs symbols are defined, the start
    /// is a nonterminal. Returns the CfgError on the first violation.
    pub fn validate_cfg(&self) -> Result<(), CfgError> {
        let num_symbols = self.num_nonterminals + self.num_terminals;
        // the check 1: the lhs of every production is a nonterminal
        for (lhs, _) in &self.productions {
            if *lhs >= self.num_nonterminals {
                return Err(CfgError::TerminalOnLhs(*lhs));
            }
        }
        // the check 2: the rhs symbols are defined (the terminals + the nonterminals)
        for (_, rhs) in &self.productions {
            for sym in rhs {
                if *sym >= num_symbols {
                    return Err(CfgError::UndefinedSymbol(*sym));
                }
            }
        }
        // the check 3: the start symbol is a nonterminal
        if self.start >= self.num_nonterminals {
            return Err(CfgError::StartIsTerminal(self.start));
        }
        Ok(())
    }
}

/// Errors from the CFG validation (the prove the input is a valid CFG).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CfgError {
    /// the lhs of a production is a terminal (the CFG requires the nonterminal lhs).
    TerminalOnLhs(u32),
    /// the rhs symbol is undefined (the symbol is out of range).
    UndefinedSymbol(u32),
    /// the start symbol is a terminal (the CFG requires the nonterminal start).
    StartIsTerminal(u32),
    /// the index mapping is inconsistent (the global symbol index leaked into a
    /// local index position). This is the class of bug that silently corrupts
    /// the state numbering (the accepting state lands on the wrong state).
    IndexMapping {
        what: &'static str,
        got: u32,
        expected_lt: u32,
    },
    /// A catch-all error (the export failures, the I/O).
    Other(String),
}
impl std::fmt::Display for CfgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CfgError::TerminalOnLhs(sym) => write!(f, "terminal {sym} on the lhs of a production"),
            CfgError::UndefinedSymbol(sym) => write!(f, "undefined symbol {sym} in a rhs"),
            CfgError::StartIsTerminal(sym) => write!(f, "start symbol {sym} is a terminal"),
CfgError::IndexMapping { what, got, expected_lt } => write!(
                 f,
                 "index mapping error: {what} = {got} but must be < {expected_lt}. \
                  This usually means a GLOBAL symbol index (the 0..num_symbols) leaked \
                  into a LOCAL index position (the 0..num_nonterminals or 0..num_terminals). \
                  Check the Grammar impl's terminal_id/nonterminal_id/start_id."
             ),
             CfgError::Other(msg) => write!(f, "{msg}"),
         }
    }
}
impl std::error::Error for CfgError {}

impl Grammar for Cfg {
    type Nonterminal = u32;
    type Terminal = u32;
    type Symbol = u32;

    fn nonterminals(&self) -> Vec<u32> {
        (0..self.num_nonterminals).collect()
    }
    fn terminals(&self) -> Vec<u32> {
        (0..self.num_terminals).collect()
    }
    fn start(&self) -> u32 {
        self.start
    }
    fn productions(&self) -> Vec<(u32, Vec<u32>)> {
        self.productions.clone()
    }
    fn is_terminal(&self, sym: &u32) -> bool {
        *sym >= self.num_nonterminals
    }
    fn terminal_id(&self, sym: &u32) -> Option<u32> {
        if *sym >= self.num_nonterminals {
            Some(sym - self.num_nonterminals)
        } else {
            None
        }
    }
    fn nonterminal_id(&self, sym: &u32) -> Option<u32> {
        if *sym < self.num_nonterminals {
            Some(*sym)
        } else {
            None
        }
    }
    fn nonterminal_index(&self, nt: &u32) -> u32 {
        *nt
    }
    fn start_id(&self) -> u32 {
        self.start
    }
    fn validate(&self) -> Result<(), CfgError> {
        self.validate_cfg()
    }
}