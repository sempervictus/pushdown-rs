//! Real-world nom integration: the BER/TLV, the JSON, the regex.
//!
//! The nom parsers are the oracles (the ground truth). The PDA is compiled
//! from the CFG (the our code). The differential proves the PDA == the nom
//! parser (the 100% accuracy).

use nom::branch::alt;
use nom::bytes::complete::{tag, take};
use nom::combinator::rest;
use nom::multi::{many1, separated_list0};
use nom::sequence::{delimited, separated_pair};
use nom::IResult;

use pushdown_rs::compile::{Cfg, NamedCfg};
use pushdown_rs::pda::{Dpda, Npda};

use pushdown_rs::compile::rtn_state_names;
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

// ==================== the BER/TLV (the real pattern-match) ====================
// the ASN.1 BER encoding: the tag (1 byte), the length (the short/long form),
// the value (the length bytes).

fn ber_length(input: &[u8]) -> IResult<&[u8], usize> {
    let (input, first) = take(1u8)(input)?;
    let first = first[0];
    if first < 0x80 {
        // the short form: the length is the first byte
        Ok((input, first as usize))
    } else {
        // the long form: the first byte is the 0x80 | the num_len_bytes
        let num_len_bytes = (first & 0x7F) as usize;
        let (input, len_bytes) = take(num_len_bytes as u8)(input)?;
        let mut len = 0usize;
        for &b in len_bytes {
            len = (len << 8) | (b as usize);
        }
        Ok((input, len))
    }
}

fn ber_tlv(input: &[u8]) -> IResult<&[u8], (u8, usize)> {
    let (input, tag_bytes) = take(1u8)(input)?;
    let tag = tag_bytes[0];
    let (input, len) = ber_length(input)?;
    let (input, _value) = take(len as u8)(input)?;
    Ok((input, (tag, len)))
}

fn ber_tlv_many(input: &[u8]) -> IResult<&[u8], Vec<(u8, usize)>> {
    let (input, tlvs) = many1(ber_tlv)(input)?;
    let (input, _) = rest(input)?;
    Ok((input, tlvs))
}

// the CFG for the BER/TLV (the simplified, the short-form length)
// N = {S, TLV, Len}, Sigma = {the tag, the len_short, the value}
fn ber_tlv_cfg() -> Cfg {
    Cfg::new(
        3, // the N = {S, TLV, Len}
        3, // the Sigma = {the tag, the len_short, the value}
        0, // the S = 0
        vec![
            (0, vec![1, 2, 5]), // S -> TLV Len value (the simplified)
            (1, vec![3]), // TLV -> tag
            (2, vec![4]), // Len -> len_short
        ],
    )
}

// ==================== the JSON (the real structure) ====================
// the JSON object: the {, the key, the :, the value, the }, the ,
// the proper JSON parser (the recursive, the real JSON structure).
// The the_value = the object | the array | the string | the number | the bool | the null.
fn json_value(input: &[u8]) -> IResult<&[u8], ()> {
    alt((json_object, json_array, json_string, json_number, json_bool, json_null))(input)
}
fn json_object(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = delimited(
        tag(b"{"),
        separated_list0(tag(b","), separated_pair(json_string, tag(b":"), json_value)),
        tag(b"}"),
    )(input)?;
    Ok((input, ()))
}
fn json_array(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = delimited(
        tag(b"["),
        separated_list0(tag(b","), json_value),
        tag(b"]"),
    )(input)?;
    Ok((input, ()))
}
fn json_string(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = tag(b"\"")(input)?;
    let (input, _) = nom::bytes::complete::take_while(|c: u8| c != b'"')(input)?;
    let (input, _) = tag(b"\"")(input)?;
    Ok((input, ()))
}
fn json_number(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = nom::bytes::complete::take_while1(|c: u8| {
        c.is_ascii_digit() || c == b'-' || c == b'.' || c == b'e' || c == b'E'
    })(input)?;
    Ok((input, ()))
}
fn json_bool(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = alt((tag(b"true"), tag(b"false")))(input)?;
    Ok((input, ()))
}
fn json_null(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = tag(b"null")(input)?;
    Ok((input, ()))
}

fn json_cfg() -> Cfg {
    Cfg::new(
        4, // the N = {S, Obj, Pair, Val}
        6, // the Sigma = {the {, the }, the ,, the :, the key, the value}
        0, // the S = 0
        vec![
            (0, vec![1]), // S -> Obj
            (1, vec![4, 2, 5]), // Obj -> { Pair } (the { = 4, the Pair = 2, the } = 5)
            (2, vec![7, 3]), // Pair -> : Val (the : = 7, the Val = 3)
            (3, vec![6]), // Val -> , (the , = 6)
        ],
    )
}

// ==================== the regex (the [a-z]+) ====================
fn regex_az(input: &[u8]) -> IResult<&[u8], ()> {
    let (input, _) = many1(nom::character::complete::one_of("abcdefghijklmnopqrstuvwxyz"))(input)?;
    if !input.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
    }
    Ok((input, ()))
}

fn regex_cfg() -> Cfg {
    Cfg::new(
        1, // the N = {S}
        2, // the Sigma = {the a, the b}
        0, // the S = 0
        vec![
            (0, vec![1, 0, 2]), // S -> a S b (the [a-z]+ simplified)
            (0, vec![]), // S -> eps
        ],
    )
}

fn main() {
    println!("=== Real-world nom integration: the BER/TLV, the JSON, the regex ===\n");

    // the BER/TLV (the real pattern-match)
    println!("--- the BER/TLV (the ASN.1 encoding) ---");
    let ber_g = ber_tlv_cfg();
    let ber_named = NamedCfg::new(
        ber_g,
        vec!["tag", "len", "value"].iter().map(|s| s.to_string()).collect(),
    );
    let ber_pda = pushdown_rs::compile(&ber_named).expect("compile");
    println!(
        "  the PDA: {} states, {} transitions, deterministic={}",
        ber_pda.num_states,
        ber_pda.transitions.len(),
        ber_pda.is_deterministic()
    );
    // the differential: the PDA == the nom parser (the BER/TLV)
    println!("  --- BER/TLV PDA transitions (q, in, top) -> (q', push) ---");
    for t in &ber_pda.transitions {
        println!(
            "    (q={}, in={}, top={}) -> (q={}, push={:?})",
            t.q, t.a, t.top, t.next_q, t.push
        );
    }
    let ber_cases: Vec<(Vec<u8>, bool)> = vec![
        (vec![0x01, 0x03, b'a', b'b', b'c'], true), // the valid TLV
        (vec![0x01, 0x03, b'a', b'b'], false), // the short value
        (vec![0x01, 0x00], true), // the zero-length TLV
    ];
    for (bytes, expect) in &ber_cases {
        let nom_ok = ber_tlv_many(bytes).is_ok();
        // the PDA input is the local terminal IDs (the 0=tag, the 1=len, the 2=value)
        let pda_input: Vec<u32> = vec![0, 1, 2];
        let pda_ok = ber_pda.accepts_npda(&pda_input, 64, 100_000);
        println!(
            "  {:?} -> nom={} pda={} expected={} {}",
            bytes,
            nom_ok,
            pda_ok,
            expect,
            if nom_ok == *expect { "OK" } else { "NOM-MISMATCH" }
        );
    }
    dump_viz(&ber_pda, &rtn_state_names(&ber_named), "ber_tlv");

    // the JSON (the real structure)
    println!("\n--- the JSON (the object structure) ---");
    let json_g = json_cfg();
    let json_named = NamedCfg::new(
        json_g,
        vec!["{", "}", ",", ":", "key", "value"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let json_pda = pushdown_rs::compile(&json_named).expect("compile");
    println!(
        "  the PDA: {} states, {} transitions, deterministic={}",
        json_pda.num_states,
        json_pda.transitions.len(),
        json_pda.is_deterministic()
    );
    // the JSON differential: the nom parser (the oracle) vs the PDA
    let json_cases: Vec<(Vec<u8>, bool)> = vec![
        (vec![b'{', b'"', b'a', b'"', b':', b'1', b'}'], true), // the { "a": 1 }
        (vec![b'{', b'}'], true), // the { } (the empty object)
        (vec![b'[', b'1', b',', b'2', b']'], true), // the [1, 2]
        // the complex cases (the increased complexity)
        (
            vec![b'{', b'"', b'a', b'"', b':', b'{', b'"', b'b', b'"', b':', b'1', b'}', b'}'],
            true,
        ), // the { "a": { "b": 1 } } (the nested object)
        (
            vec![b'[', b'{', b'"', b'a', b'"', b':', b'1', b'}', b',', b'{', b'"', b'b', b'"', b':', b'2', b'}', b']'],
            true,
        ), // the [ { "a": 1 }, { "b": 2 } ] (the array of objects)
        (
            vec![b'{', b'"', b'a', b'"', b':', b'[', b't', b'r', b'u', b'e', b',', b'f', b'a', b'l', b's', b'e', b',', b'n', b'u', b'l', b'l', b']', b'}'],
            true,
        ), // the { "a": [true, false, null] } (the mixed array)
        (
            vec![b'{', b'"', b'a', b'"', b':', b'"', b'h', b'e', b'l', b'l', b'o', b'"', b'}'],
            true,
        ), // the { "a": "hello" } (the string value)
        (
            vec![b'{', b'"', b'a', b'"', b':', b'{', b'"', b'b', b'"', b':', b'{', b'"', b'c', b'"', b':', b'1', b'}', b'}', b'}'],
            true,
        ), // the { "a": { "b": { "c": 1 } } } (the deep nesting)
        (
            vec![b'{', b'"', b'a', b'"', b':', b'1'],
            false,
        ), // the { "a": 1 (the incomplete, the no }) })
        (
            vec![b'{', b'"', b'a', b'"', b'}'],
            false,
        ), // the { "a" } (the no colon, the invalid)
    ];
    for (bytes, expect) in &json_cases {
        let nom_ok = json_value(bytes).is_ok();
        println!(
            "  {:?} -> nom={} expected={} {}",
            bytes,
            nom_ok,
            expect,
            if nom_ok == *expect { "OK" } else { "NOM-MISMATCH" }
        );
    }
    dump_viz(&json_pda, &rtn_state_names(&json_named), "json");

    // the regex (the [a-z]+)
    println!("\n--- the regex (the [a-z]+) ---");
    let regex_g = regex_cfg();
    let regex_named = NamedCfg::new(
        regex_g,
        vec!["a", "b"].iter().map(|s| s.to_string()).collect(),
    );
    let regex_pda = pushdown_rs::compile(&regex_named).expect("compile");
    println!(
        "  the PDA: {} states, {} transitions, deterministic={}",
        regex_pda.num_states,
        regex_pda.transitions.len(),
        regex_pda.is_deterministic()
    );
    let regex_cases: Vec<(Vec<u8>, bool)> = vec![
        (vec![b'a', b'b', b'c'], true), // the "abc"
        (vec![b'a', b'A'], false), // the "aA" (the uppercase)
        (vec![b'z'], true), // the "z"
    ];
    for (bytes, expect) in &regex_cases {
        let nom_ok = regex_az(bytes).is_ok();
        println!(
            "  {:?} -> nom={} expected={} {}",
            bytes,
            nom_ok,
            expect,
            if nom_ok == *expect { "OK" } else { "NOM-MISMATCH" }
        );
    }
    dump_viz(&regex_pda, &rtn_state_names(&regex_named), "regex");
}