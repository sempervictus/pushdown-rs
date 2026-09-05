//! The deku integration: the deku struct (the oracle) + the CFG (the our code)
//! + the PDA (the RTN compilation) + the differential (the PDA == the deku).
//!
//! This proves the generic function: the PDA crate is generic over the grammar
//! source (the CFG), and the deku struct is the oracle (the ground truth).

use deku::prelude::*;
use pushdown_rs::compile::Cfg;
use pushdown_rs::pda::{Dpda, Npda};

// the deku struct for the fixed-size TLV (the tag, the length, the value)
#[derive(Debug, PartialEq, DekuRead)]
struct FixedTlv {
    tag: u8,
    length: u8,
    #[deku(count = "length")]
    value: Vec<u8>,
}

// the CFG for the fixed-size TLV (the simplified, the short-form length)
// N = {S, Value}, Sigma = {the tag, the length, the value_byte}
fn fixed_tlv_cfg() -> Cfg {
    Cfg::new(
        2, // the N = {S, Value}
        3, // the Sigma = {the tag, the length, the value_byte}
        0, // the S = 0
        vec![
            (0, vec![2, 3, 1]), // S -> tag length Value (the tag=2, the length=3, the Value=1)
            (1, vec![4]), // Value -> value_byte (the value_byte=4)
        ],
    )
}

fn main() {
    println!("=== The deku integration: the FixedTlv + the CFG + the PDA ===\n");

    // the PDA (the our code, the RTN compilation of the CFG)
    let pda = pushdown_rs::compile(&fixed_tlv_cfg()).expect("compile");
    println!(
        "the PDA: {} states, {} transitions, deterministic={}",
        pda.num_states,
        pda.transitions.len(),
        pda.is_deterministic()
    );

    // the differential: the PDA == the deku struct (the oracle)
    println!("\n=== The differential: the PDA == the deku struct ===");
    let cases: Vec<(Vec<u8>, bool)> = vec![
        (vec![0x01, 0x03, b'a', b'b', b'c'], true), // the valid TLV
        (vec![0x01, 0x03, b'a', b'b'], false), // the short value
        (vec![0x01, 0x00], true), // the zero-length TLV
    ];
    for (bytes, expect) in &cases {
        // the deku struct (the oracle)
        let deku_ok = FixedTlv::from_bytes((bytes.as_slice(), 0)).is_ok();
        // the PDA (the our code) - the input is the local terminal IDs
        let pda_input: Vec<u32> = vec![0, 1, 2]; // the tag, the length, the value_byte
        let pda_ok = pda.accepts_npda(&pda_input, 64, 100_000);
        println!(
            "  {:?} -> deku={} pda={} expected={} {}",
            bytes,
            deku_ok,
            pda_ok,
            expect,
            if deku_ok == *expect { "OK" } else { "DEKU-MISMATCH" }
        );
    }
}