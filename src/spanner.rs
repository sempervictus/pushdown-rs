//! The token spanner (the GreatGramma T_inv, arXiv:2502.05111).
//!
//! The token spanner maps (the lexer state, the terminal sequence) to the set
//! of tokens that produce that sequence from that state. This is the bridge
//! between the PDA (the terminal-level) and the tokenizer (the token-level).
//!
//! The T_inv is the precomputed source data for the GPU/SIMD construction: the
//! mask at a PDA config is the T_inv over the accepted terminal sequences.

use std::collections::HashMap;

/// The tokenizer: the token -> the bytes (the P4 primitive).
pub trait Tokenizer {
    type Token: Copy + Eq + std::hash::Hash;
    fn token_bytes(&self, t: Self::Token) -> Vec<u8>;
    fn vocab(&self) -> Vec<Self::Token>;
}

/// The lexer DFA (the P3 primitive): the byte-level state machine.
pub trait LexerDfa {
    type LState: Copy + Eq + std::hash::Hash;
    type Terminal: Copy + Eq + std::hash::Hash;
    fn transition(&self, s: Self::LState, b: u8) -> Self::LState;
    fn is_dead(&self, s: Self::LState) -> bool;
    /// the terminals that COMPLETE at this state (the accepting set).
    fn accepting(&self, s: Self::LState) -> Vec<Self::Terminal>;
    fn initial(&self) -> Self::LState;
}

/// The token spanner (the T_inv): the (q_lex, terminal sequence) -> the tokens
/// that produce the sequence from the state.
pub struct TokenSpanner<L: LexerDfa, T: Tokenizer> {
    // the forward: the (q_lex, token) -> the terminal sequence
    forward: HashMap<(L::LState, T::Token), Vec<L::Terminal>>,
    // the inverse: the (q_lex, the terminal sequence) -> the tokens
    inverse: HashMap<(L::LState, Vec<L::Terminal>), Vec<T::Token>>,
}

impl<L: LexerDfa, T: Tokenizer> TokenSpanner<L, T> {
    /// Build the token spanner (the offline preprocessing): drive each DFA over
    /// the vocab for each reachable q_lex, recording the (q_lex, token) ->
    /// sequence + the inverse (q_lex, sequence) -> tokens.
    pub fn build(dfa: &L, tok: &T, q_lex_states: &[L::LState]) -> Self {
        let mut forward: HashMap<(L::LState, T::Token), Vec<L::Terminal>> = HashMap::new();
        let mut inverse: HashMap<(L::LState, Vec<L::Terminal>), Vec<T::Token>> = HashMap::new();
        for &q_lex in q_lex_states {
            for &token in &tok.vocab() {
                let bytes = tok.token_bytes(token);
                let seq = Self::drive(dfa, q_lex, &bytes);
                forward.insert((q_lex, token), seq.clone());
                inverse.entry((q_lex, seq.clone())).or_default().push(token);
            }
        }
        TokenSpanner { forward, inverse }
    }

    /// Drive the DFA over the token's bytes: the terminal sequence produced.
    fn drive(dfa: &L, q_lex: L::LState, bytes: &[u8]) -> Vec<L::Terminal> {
        let mut cur = q_lex;
        let mut seq = Vec::new();
        for &b in bytes {
            cur = dfa.transition(cur, b);
            if dfa.is_dead(cur) {
                break;
            }
            seq.extend(dfa.accepting(cur));
        }
        seq
    }

    /// The T_inv: the tokensq_lex, the terminal sequence) -> the tokens that
    /// produce the sequence from the state.
    pub fn tokens_for(&self, q_lex: L::LState, seq: &[L::Terminal]) -> &[T::Token] {
        self.inverse
            .get(&(q_lex, seq.to_vec()))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// The forward: the (q_lex, token) -> the terminal sequence.
    pub fn sequence_of(&self, q_lex: L::LState, token: T::Token) -> &[L::Terminal] {
        self.forward
            .get(&(q_lex, token))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}