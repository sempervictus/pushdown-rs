//! PDA -> SVG / DOT visualization (a pure diagnostic view over the machine).
//!
//! This module renders the control-state graph of a [`PdaMachine`] as a
//! human-meaningful picture:
//! - states are nodes (the start is ringed green, the accepting set is filled);
//! - transitions are labeled edges (the input symbol + the stack op);
//! - the layout is a left-to-right column flow (BFS depth from the start).
//!
//! It is a **view, not an execution tier**: it does not enter the
//! scalar -> SIMD -> GPU identity chain (the AGENTS.md device-parity invariant),
//! and it is a pure function of the already-validated machine. The rendering
//! invariants are:
//!   - one primary node per control state (the node count == `num_states`);
//!   - one edge per transition (the edge count == `transitions.len()`);
//!   - epsilon moves are dashed + red, terminal moves are solid + grey;
//!   - the stack op is annotated (`pop` / `keep` / `push(k)`).
//!
//! Two emitters:
//! - [`to_svg`] - a self-contained SVG (zero deps, no-unsafe). The primary.
//! - [`to_dot`] - a Graphviz DOT file (run `dot -Tsvg` for a prettier auto-layout).
//!
//! Convenience file writers: [`write_svg`], [`write_dot`], [`dump_g`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::path::Path;

use crate::compile::Grammar;
use crate::machine::{PdaMachine, Transition};
use crate::pda::Dpda;

// The layout constants (the pixels).
const MARGIN: f64 = 64.0;
const COL_W: f64 = 210.0;
const ROW_H: f64 = 92.0;
const NODE_R: f64 = 26.0;
const FONT: f64 = 12.0;

/// One node's screen position.
#[derive(Debug, Clone, Copy)]
struct Pos {
    x: f64,
    y: f64,
}

/// Compute the column (BFS depth) of every state, then assign a (x, y).
///
/// Reachable states flow left -> right by depth; unreachable states are parked
/// in a trailing column (the max reached depth + 1) so they are still drawn.
/// Within a column, states are ordered by id (deterministic).
fn layout(m: &PdaMachine) -> Vec<Pos> {
    let n = m.num_states as usize;
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for t in &m.transitions {
        adj[t.q as usize].push(t.next_q as usize);
    }
    let mut depth = vec![usize::MAX; n];
    depth[m.start_state as usize] = 0;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(m.start_state as usize);
    while let Some(u) = q.pop_front() {
        for &v in &adj[u] {
            if depth[v] == usize::MAX {
                depth[v] = depth[u] + 1;
                q.push_back(v);
            }
        }
    }
    let max_reached = depth.iter().copied().filter(|d| *d != usize::MAX).max().unwrap_or(0);
    for d in depth.iter_mut() {
        if *d == usize::MAX {
            *d = max_reached + 1;
        }
    }
    let mut cols: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, d) in depth.iter().enumerate() {
        cols.entry(*d).or_default().push(i);
    }
    let mut pos = vec![Pos { x: 0.0, y: 0.0 }; n];
    for (col, members) in cols.iter() {
        for (row, &sid) in members.iter().enumerate() {
            pos[sid] = Pos {
                x: MARGIN + (*col as f64) * COL_W,
                y: MARGIN + (row as f64) * ROW_H,
            };
        }
    }
    pos
}

/// The state's label: the caller-supplied name if present, else `q<id>`.
fn state_label(_m: &PdaMachine, id: u32, names: Option<&[String]>) -> String {
    match names {
        Some(n) if (id as usize) < n.len() => n[id as usize].clone(),
        _ => format!("q{id}"),
    }
}

/// The input's label: `epsilon` for the epsilon move (a == num_inputs), else the
/// caller-supplied terminal name if present, else the raw id.
fn input_label(m: &PdaMachine, a: u32, terms: Option<&[String]>) -> String {
    if a == m.num_inputs {
        "epsilon".to_string()
    } else {
        match terms {
            Some(t) if (a as usize) < t.len() => t[a as usize].clone(),
            _ => format!("{a}"),
        }
    }
}

/// The stack op of a transition: `pop` (empty push), `keep` (push == [top]),
/// or `push(k)` (k symbols pushed, the top of the push is the new top).
fn stack_op_label(t: &Transition) -> String {
    match t.push.len() {
        0 => "pop".to_string(),
        1 if t.push[0] == t.top => "keep".to_string(),
        k => format!("push({k})"),
    }
}

/// XML-escape a string for use inside an SVG text/attribute node.
fn xml_esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// DOT-escape a string for use inside a quoted DOT identifier/label.
fn dot_esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Render the machine as a self-contained SVG string (the primary emitter).
///
/// `state_names` / `term_names` are optional label arrays (the
/// [`rtn_state_names`](crate::compile::rtn_state_names) for RTN machines, the
/// terminal vocabulary for the inputs). When absent, the raw ids are shown.
pub fn to_svg(
    m: &PdaMachine,
    state_names: Option<&[String]>,
    term_names: Option<&[String]>,
) -> String {
    let pos = layout(m);
    let n = m.num_states as usize;
    let accepting: HashSet<u32> = m.accepting.iter().copied().collect();

    let max_x = pos.iter().map(|p| p.x).fold(0.0, f64::max) + MARGIN + NODE_R + 40.0;
    let max_y = pos.iter().map(|p| p.y).fold(0.0, f64::max) + MARGIN + NODE_R + 40.0;

    // Group parallel edges (the same (q, next_q)) so they fan out perpendicular.
    let mut groups: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, t) in m.transitions.iter().enumerate() {
        groups.entry((t.q as usize, t.next_q as usize)).or_default().push(i);
    }
    // Sort the groups so the rendered edge order is deterministic (the pure view).
    let mut group_keys: Vec<(usize, usize)> = groups.keys().cloned().collect();
    group_keys.sort();

    let mut s = String::new();
    s.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{max_x:.0}" height="{max_y:.0}" viewBox="0 0 {max_x:.0} {max_y:.0}" font-family="monospace" font-size="{FONT:.0}">
  <title>PDA: {n} states, {t} transitions, deterministic={det}</title>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker>
    <marker id="arrE" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#b34"/></marker>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
"##,
        max_x = max_x,
        max_y = max_y,
        n = n,
        t = m.transitions.len(),
        det = m.is_deterministic(),
    ));

    // The edges (drawn first, under the nodes).
    for (q, nq) in group_keys.iter() {
        let idxs = &groups[&((*q), (*nq))];
        let self_loop = q == nq;
        for (k, &ti) in idxs.iter().enumerate() {
            let t = &m.transitions[ti];
            let is_eps = t.a == m.num_inputs;
            let a = input_label(m, t.a, term_names);
            let op = stack_op_label(t);
            let label = if is_eps {
                format!("epsilon . {op}")
            } else {
                format!("{a} . {op}")
            };
            let color = if is_eps { "#b34" } else { "#555" };
            let marker = if is_eps { "arrE" } else { "arr" };
            let dash = if is_eps { r##" stroke-dasharray="5,4""## } else { "" };

            if self_loop {
                let p = pos[*q];
                let cx = p.x;
                let cy = p.y - NODE_R - 16.0;
                s.push_str(&format!(
                    r##"  <path d="M {cx:.1} {cy:.1} a 15 15 0 1 1 0.1 0" fill="none" stroke="{color}"{dash} marker-end="url(#{marker})"/>
  <text x="{cx:.1}" y="{ty:.1}" fill="{color}" text-anchor="middle" font-size="10">{lab}</text>
"##,
                    cx = cx,
                    cy = cy,
                    color = color,
                    dash = dash,
                    marker = marker,
                    ty = cy - 18.0,
                    lab = xml_esc(&label),
                ));
            } else {
                let p1 = pos[*q];
                let p2 = pos[*nq];
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let len = (dx * dx + dy * dy).sqrt().max(1.0);
                let ux = dx / len;
                let uy = dy / len;
                let px = -uy;
                let py = ux;
                let off = (k as f64 - (idxs.len() as f64 - 1.0) / 2.0) * 9.0;
                let sx = p1.x + ux * NODE_R + px * off;
                let sy = p1.y + uy * NODE_R + py * off;
                let ex = p2.x - ux * NODE_R + px * off;
                let ey = p2.y - uy * NODE_R + py * off;
                let mx = (sx + ex) / 2.0;
                let my = (sy + ey) / 2.0 - 4.0;
                s.push_str(&format!(
                    r##"  <line x1="{sx:.1}" y1="{sy:.1}" x2="{ex:.1}" y2="{ey:.1}" stroke="{color}" stroke-width="1.4"{dash} marker-end="url(#{marker})"/>
  <text x="{mx:.1}" y="{my:.1}" fill="{color}" text-anchor="middle" font-size="10">{lab}</text>
"##,
                    sx = sx,
                    sy = sy,
                    ex = ex,
                    ey = ey,
                    color = color,
                    dash = dash,
                    marker = marker,
                    mx = mx,
                    my = my,
                    lab = xml_esc(&label),
                ));
            }
        }
    }

    // The nodes (drawn last, over the edges).
    for (i, p) in pos.iter().enumerate() {
        let is_start = i == m.start_state as usize;
        let is_acc = accepting.contains(&(i as u32));
        let fill = if is_acc { "#fde" } else { "#fff" };
        let stroke = if is_start { "#0a0" } else { "#333" };
        let sw = if is_start { "3" } else { "1.5" };
        s.push_str(&format!(
            r##"  <circle cx="{x:.1}" cy="{y:.1}" r="{r:.0}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>
"##,
            x = p.x,
            y = p.y,
            r = NODE_R,
            fill = fill,
            stroke = stroke,
            sw = sw,
        ));
        if is_start {
            s.push_str(&format!(
                r##"  <circle cx="{x:.1}" cy="{y:.1}" r="{r:.0}" fill="none" stroke="#0a0" stroke-width="1.5"/>
"##,
                x = p.x,
                y = p.y,
                r = NODE_R + 5.0,
            ));
        }
        let name = state_label(m, i as u32, state_names);
        s.push_str(&format!(
            r##"  <text x="{x:.1}" y="{y:.1}" text-anchor="middle" fill="#111">{lab}</text>
"##,
            x = p.x,
            y = p.y + 4.0,
            lab = xml_esc(&name),
        ));
        let tag = if is_start && is_acc {
            "start+accept"
        } else if is_start {
            "start"
        } else if is_acc {
            "accept"
        } else {
            ""
        };
        if !tag.is_empty() {
            s.push_str(&format!(
                r##"  <text x="{x:.1}" y="{y:.1}" text-anchor="middle" fill="#666" font-size="9">{tag}</text>
"##,
                x = p.x,
                y = p.y + NODE_R + 14.0,
                tag = tag,
            ));
        }
    }

    s.push_str("</svg>\n");
    s
}

/// Render the machine as a Graphviz DOT string (the secondary emitter).
///
/// Run `dot -Tsvg machine.dot -o machine.svg` for a prettier auto-layout.
pub fn to_dot(
    m: &PdaMachine,
    state_names: Option<&[String]>,
    term_names: Option<&[String]>,
) -> String {
    let accepting: HashSet<u32> = m.accepting.iter().copied().collect();
    let mut s = String::from("digraph PDA {\n  node [shape=ellipse, fontname=monospace];\n");
    for i in 0..m.num_states {
        let name = state_label(m, i, state_names);
        let mut attrs: Vec<String> = Vec::new();
        if i == m.start_state {
            attrs.push("color=green".to_string());
            attrs.push("penwidth=2".to_string());
        }
        if accepting.contains(&i) {
            attrs.push("fillcolor=lightgrey".to_string());
            attrs.push("style=filled".to_string());
        }
        let attr = if attrs.is_empty() {
            String::new()
        } else {
            format!(" [{}]", attrs.join(", "))
        };
        s.push_str(&format!("  \"{}\"{};\n", dot_esc(&name), attr));
    }
    for t in &m.transitions {
        let a = input_label(m, t.a, term_names);
        let op = stack_op_label(t);
        let label = format!("{a} / {op}");
        let style = if t.a == m.num_inputs {
            ", style=dashed, color=red"
        } else {
            ""
        };
        s.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"{}];\n",
            dot_esc(&state_label(m, t.q, state_names)),
            dot_esc(&state_label(m, t.next_q, state_names)),
            dot_esc(&label),
            style
        ));
    }
    s.push_str("}\n");
    s
}

/// Write [`to_svg`] to a file (the convenience writer).
pub fn write_svg(
    m: &PdaMachine,
    state_names: Option<&[String]>,
    term_names: Option<&[String]>,
    path: &Path,
) -> std::io::Result<()> {
    let svg = to_svg(m, state_names, term_names);
    let mut f = std::fs::File::create(path)?;
    f.write_all(svg.as_bytes())?;
    f.flush()?;
    Ok(())
}

/// Write [`to_dot`] to a file (the convenience writer).
pub fn write_dot(
    m: &PdaMachine,
    state_names: Option<&[String]>,
    term_names: Option<&[String]>,
    path: &Path,
) -> std::io::Result<()> {
    let dot = to_dot(m, state_names, term_names);
    let mut f = std::fs::File::create(path)?;
    f.write_all(dot.as_bytes())?;
    f.flush()?;
    Ok(())
}

/// A ready-to-render SVG for a grammar-compiled machine: it pairs the state
/// names from [`rtn_state_names`](crate::compile::rtn_state_names) with the
/// terminal names from the caller, then writes both the `.svg` and the `.dot`
/// to `dir`. Returns the two paths.
pub fn dump_g(
    g: &impl Grammar,
    m: &PdaMachine,
    term_names: &[&str],
    dir: &Path,
) -> std::io::Result<(std::path::PathBuf, std::path::PathBuf)> {
    use crate::compile::rtn_state_names;
    std::fs::create_dir_all(dir)?;
    let states = rtn_state_names(g);
    let terms: Vec<String> = term_names.iter().map(|s| s.to_string()).collect();
    let base = dir.join(format!("pda_{}_states", m.num_states));
    let svg_path = base.with_extension("svg");
    let dot_path = base.with_extension("dot");
    write_svg(m, Some(&states), Some(&terms), &svg_path)?;
    write_dot(m, Some(&states), Some(&terms), &dot_path)?;
    Ok((svg_path, dot_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Cfg, kappa};

    fn anb() -> (Cfg, PdaMachine) {
        let g = Cfg::new(
            1,
            2,
            0,
            vec![
                (0, vec![1, 0, 2]), // S -> a S b
                (0, vec![]), // S -> eps
            ],
        );
        let m = crate::compile(&g).expect("compile");
        (g, m)
    }

    #[test]
    fn svg_renders_every_state_and_transition() {
        let (g, m) = anb();
        let svg = to_svg(
            &m,
            Some(&crate::compile::rtn_state_names(&g)),
            Some(&["a".into(), "b".into()]),
        );
        // The edge invariant: one marker-end per transition.
        assert_eq!(svg.matches("marker-end=").count(), m.transitions.len());
        // The node invariant: at least one primary circle per state (the start
        // adds a second ring, so >= num_states).
        assert!(svg.matches("<circle").count() >= m.num_states as usize);
        // The title carries the machine summary (the human-meaningful header).
        assert!(svg.contains(&format!("PDA: {} states", m.num_states)));
    }

    #[test]
    fn dot_renders_every_state_and_transition() {
        let (g, m) = anb();
        let dot = to_dot(&m, Some(&crate::compile::rtn_state_names(&g)), None);
        assert_eq!(dot.matches(" -> ").count(), m.transitions.len());
        // One node line per state (the node lines do NOT contain " -> "; the edge lines do).
        let node_lines = dot
            .lines()
            .filter(|l| {
                let t = l.trim_start();
                t.starts_with('"') && t.contains(';') && !t.contains(" -> ")
            })
            .count();
        assert_eq!(node_lines, m.num_states as usize);
    }

    #[test]
    fn svg_is_deterministic() {
        let (_g, m) = anb();
        let a = to_svg(&m, None, None);
        let b = to_svg(&m, None, None);
        assert_eq!(a, b, "the same machine must render identically (the pure view)");
    }

    #[test]
    fn kappa_matches_rendered_node_count() {
        let (g, m) = anb();
        assert_eq!(
            m.num_states,
            kappa(&g),
            "the rendered node count is the exact kappa(G)"
        );
    }
}