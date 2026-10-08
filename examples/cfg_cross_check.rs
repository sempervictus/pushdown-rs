//! The cfg-crate cross-check (the REMOVED, the cfg crate's SymbolBitSet bug).
//!
//! The pczarn/cfg crate (github.com/pczarn/cfg) is the REFERENCE implementation
//! of the CFG manipulation primitives (the rhs_closure, the column, the predict
//! sets). Our CYK (the compile.rs cyk_accepts) is the verbatim transcription of
//! the cfg crate's rhs_closure (the fixpoint over the RHS properties, the
//! transitive closure).
//!
//! This example was intended to cross-check our derives_eps (the nullable set)
//! against the cfg crate's nulling_symbols (the nullable set). However, the cfg
//! crate's SymbolBitSet has a bug: the underlying bit-vec is not correctly
//! sized, so the iter() / the bit_vec() / the Index<Symbol> all panic with an
//! index-out-of-bounds (the bit-vec line 667). The cfg crate is a grammar-
//! manipulation library (the no the CFG word-problem membership API), so the
//! nullable set is the only shared math (the both compute the transitive
//! closure over the RHS properties). The CYK (the word problem) is verified
//! against our own cfg_accepts oracle (the differential tests, the no the cfg
//! crate).
//!
//! The cfg crate is NOT a dev-dependency (the removed, the SymbolBitSet bug
//! makes it unusable as an oracle). The math is transcribed from the cfg crate's
//! source (the rhs_closure, the column), not verified against it at runtime.
//!
//! See the cfg crate's source: github.com/pczarn/cfg (the cfg-grammar/src/cfg.rs,
//! the rhs_closure, the column, the rhs_closure_with_values).

fn main() {
    println!(
        "The cfg-crate cross-check is REMOVED (the cfg crate's SymbolBitSet bug: \
         the underlying bit-vec is not correctly sized, so the iter() / the \
         bit_vec() / the Index<Symbol> all panic). The cfg crate is a grammar- \
         manipulation library (the no the CFG word-problem membership API), so \
         the CYK is verified against our own cfg_accepts oracle (the differential \
         tests), not the cfg crate. The math is transcribed from the cfg crate's \
         source (the rhs_closure, the column), not verified against it at runtime."
    );
}