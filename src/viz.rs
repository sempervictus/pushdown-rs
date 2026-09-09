//! PDA -> SVG / DOT visualization (a pure diagnostic view over the machine).
//!
//! This module renders the control-state graph of a [`PdaMachine`] as a
//! human-meaningful picture:
//! - states are rounded-rectangle nodes (the start is ringed green, the
//!   accepting set is filled blue);
//! - transitions are curved, labeled edges (the input symbol + the stack op);
//! - the layout is a left-to-right column flow (BFS depth from the start),
//!   with each column vertically centered.
//!
//! It is a **view, not an execution tier**: it does not enter the
//! scalar -> SIMD -> GPU identity chain (the AGENTS.md device-parity invariant),
//! and it is a pure function of the already-validated machine. The rendering
//! invariants are:
//!   - one primary node per control state (the node count == `num_states`);
//!   - one edge per transition (the edge count == `transitions.len()`);
//!   - epsilon moves are dashed + amber, terminal moves are solid + slate;
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
const MARGIN_X: f64 = 48.0;
const MARGIN_TOP: f64 = 48.0;
const HEADER_H: f64 = 64.0;
const COL_W: f64 = 232.0;
const ROW_H: f64 = 108.0;
const NODE_W: f64 = 132.0;
const NODE_H: f64 = 48.0;
const FONT: f64 = 13.0;

/// One node's screen position (the center of the rounded rectangle).
#[derive(Debug, Clone, Copy)]
struct Pos {
    x: f64,
    y: f64,
}

/// Compute the column (BFS depth) of every state, then assign a (x, y) center.
///
/// Reachable states flow left -> right by depth; unreachable states are parked
/// in a trailing column (the max reached depth + 1) so they are still drawn.
/// Within a column, states are ordered by id (deterministic) and the column is
/// vertically centered against the tallest column (the balanced look).
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
    let num_cols = cols.len();
    let max_rows = cols.values().map(|v| v.len()).max().unwrap_or(1);
    // Density-aware spacing: wide graphs get more column room, tall columns get
    // more row room, so the edge bundles (the parallel edges) do not pile up.
    let col_w = COL_W + (num_cols as f64 - 1.0) * 12.0;
    let row_h = ROW_H + (max_rows as f64 - 1.0) * 10.0;
    let mut pos = vec![Pos { x: 0.0, y: 0.0 }; n];
    for (col, members) in cols.iter() {
        let top_offset = (max_rows - members.len()) / 2;
        for (i, &sid) in members.iter().enumerate() {
            let row = top_offset + i;
            pos[sid] = Pos {
                x: MARGIN_X + *col as f64 * col_w + NODE_W / 2.0,
                y: HEADER_H + MARGIN_TOP + row as f64 * row_h + NODE_H / 2.0,
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
/// caller-supplied terminal name if present, else the machine's own vocabulary
/// (the `vocab_names`, the CPU-side labels), else the raw id.
fn input_label(m: &PdaMachine, a: u32, terms: Option<&[String]>) -> String {
    if a == m.num_inputs {
        return "epsilon".to_string();
    }
    if let Some(t) = terms {
        if (a as usize) < t.len() {
            return t[a as usize].clone();
        }
    }
    // Fall back to the machine's own vocabulary (the vocab_names, the
    // terminal_name the grammar supplied). This is the single source of truth:
    // a compiled machine renders its real symbols with no caller-side labels.
    m.vocab_name(a)
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{a}"))
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

    let max_x = pos.iter().map(|p| p.x).fold(0.0, f64::max) + NODE_W / 2.0 + MARGIN_X;
    let max_y = pos.iter().map(|p| p.y).fold(0.0, f64::max) + NODE_H / 2.0 + MARGIN_TOP + 30.0;

    // Group parallel edges (the same (q, next_q)) so they fan out vertically.
    let mut groups: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, t) in m.transitions.iter().enumerate() {
        groups.entry((t.q as usize, t.next_q as usize)).or_default().push(i);
    }
    // Sort the groups so the rendered edge order is deterministic (the pure view).
    let mut group_keys: Vec<(usize, usize)> = groups.keys().cloned().collect();
    group_keys.sort();

    let det = m.is_deterministic();
    let mut s = String::new();
    s.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{max_x:.0}" height="{max_y:.0}" viewBox="0 0 {max_x:.0} {max_y:.0}" font-family="'Inter','Segoe UI',system-ui,-apple-system,sans-serif">
  <title>PDA: {n} states, {t} transitions, deterministic={det}</title>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker>
    <marker id="arrE" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#f59e0b"/></marker>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{mx}" y="30" font-size="17" font-weight="600" fill="#0f172a">PDA &#8212; {n} states &#183; {t} transitions &#183; deterministic={det}</text>
  <text x="{mx}" y="48" font-size="11" fill="#64748b">solid slate = terminal move &#183; dashed amber = epsilon &#183; label = input / stack-op</text>
"##,
        max_x = max_x,
        max_y = max_y,
        n = n,
        t = m.transitions.len(),
        det = det,
        mx = MARGIN_X,
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
            let a_esc = xml_esc(&a);
            let a_disp = if is_eps { "&#949;" } else { a_esc.as_str() };
            let label = format!("{a_disp} &#183; {op}");
            let color = if is_eps { "#f59e0b" } else { "#94a3b8" };
            let marker = if is_eps { "arrE" } else { "arr" };
            let dash = if is_eps { r##" stroke-dasharray="6,4""## } else { "" };

            if self_loop {
                let p = pos[*q];
                let cx = p.x;
                let top = p.y - NODE_H / 2.0;
                s.push_str(&format!(
                    r##"  <path d="M {x1:.1} {y1:.1} C {c1x:.1} {c1y:.1}, {c2x:.1} {c2y:.1}, {x2:.1} {y2:.1}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>
  <text x="{lx:.1}" y="{ly:.1}" fill="{color}" font-size="10" text-anchor="middle" paint-order="stroke" stroke="#ffffff" stroke-width="3">{lab}</text>
"##,
                    x1 = cx - 24.0,
                    y1 = top,
                    c1x = cx - 58.0,
                    c1y = top - 46.0,
                    c2x = cx + 58.0,
                    c2y = top - 46.0,
                    x2 = cx + 24.0,
                    y2 = top,
                    color = color,
                    dash = dash,
                    marker = marker,
                    lx = cx,
                    ly = top - 42.0,
                    lab = label,
                ));
            } else {
                let p1 = pos[*q];
                let p2 = pos[*nq];
                let forward = p2.x >= p1.x;
                let sx = if forward {
                    p1.x + NODE_W / 2.0
                } else {
                    p1.x - NODE_W / 2.0
                };
                let ex = if forward {
                    p2.x - NODE_W / 2.0
                } else {
                    p2.x + NODE_W / 2.0
                };
                let off = (k as f64 - (idxs.len() as f64 - 1.0) / 2.0) * 14.0;
                let sy = p1.y + off;
                let ey = p2.y + off;
                let c1x = sx + (ex - sx) * 0.5;
                let c2x = sx + (ex - sx) * 0.5;
                let mx = (sx + ex) / 2.0;
                let my = (sy + ey) / 2.0 - 6.0;
                s.push_str(&format!(
                    r##"  <path d="M {sx:.1} {sy:.1} C {c1x:.1} {sy:.1}, {c2x:.1} {ey:.1}, {ex:.1} {ey:.1}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>
  <text x="{mx:.1}" y="{my:.1}" fill="{color}" font-size="10" text-anchor="middle" paint-order="stroke" stroke="#ffffff" stroke-width="3">{lab}</text>
"##,
                    sx = sx,
                    sy = sy,
                    c1x = c1x,
                    c2x = c2x,
                    ex = ex,
                    ey = ey,
                    color = color,
                    dash = dash,
                    marker = marker,
                    mx = mx,
                    my = my,
                    lab = label,
                ));
            }
        }
    }

    // The nodes (drawn last, over the edges).
    for (i, p) in pos.iter().enumerate() {
        let is_start = i == m.start_state as usize;
        let is_acc = accepting.contains(&(i as u32));
        let (fill, stroke, sw) = if is_acc {
            ("#eff6ff", "#3b82f6", "2.0")
        } else {
            ("#ffffff", "#cbd5e1", "1.5")
        };
        let (fill, stroke, sw) = if is_start {
            ("#ecfdf5", "#10b981", "2.5")
        } else {
            (fill, stroke, sw)
        };
        let x = p.x - NODE_W / 2.0;
        let y = p.y - NODE_H / 2.0;
        s.push_str(&format!(
            r##"  <rect x="{x:.1}" y="{y:.1}" width="{w:.0}" height="{h:.0}" rx="10" ry="10" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>
"##,
            x = x,
            y = y,
            w = NODE_W,
            h = NODE_H,
            fill = fill,
            stroke = stroke,
            sw = sw,
        ));
        let name = state_label(m, i as u32, state_names);
        s.push_str(&format!(
            r##"  <text x="{cx:.1}" y="{cy:.1}" text-anchor="middle" font-size="{fs:.0}" font-weight="500" fill="#1e293b">{lab}</text>
"##,
            cx = p.x,
            cy = p.y + 5.0,
            fs = FONT,
            lab = xml_esc(&name),
        ));
        let tag = if is_start && is_acc {
            "start &#183; accept"
        } else if is_start {
            "start"
        } else if is_acc {
            "accept"
        } else {
            ""
        };
        if !tag.is_empty() {
            s.push_str(&format!(
                r##"  <text x="{cx:.1}" y="{cy:.1}" text-anchor="middle" font-size="9" fill="#94a3b8" letter-spacing="0.5">{lab}</text>
"##,
                cx = p.x,
                cy = p.y + NODE_H / 2.0 + 14.0,
                lab = tag,
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
    let mut s = String::from(
        "digraph PDA {\n  graph [rankdir=LR, splines=spline, nodesep=0.5, ranksep=0.9];\n  node [shape=box, style=rounded, fontname=\"Helvetica,Arial,sans-serif\", fontsize=12, fillcolor=white, color=#cbd5e1];\n",
    );
    for i in 0..m.num_states {
        let name = state_label(m, i, state_names);
        let mut attrs: Vec<String> = Vec::new();
        if i == m.start_state {
            attrs.push("color=#10b981".to_string());
            attrs.push("penwidth=2".to_string());
            attrs.push("fillcolor=#ecfdf5".to_string());
            attrs.push("style=rounded,filled".to_string());
        }
        if accepting.contains(&i) {
            attrs.push("fillcolor=#eff6ff".to_string());
            attrs.push("color=#3b82f6".to_string());
            attrs.push("style=rounded,filled".to_string());
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
            ", style=dashed, color=#f59e0b"
        } else {
            ", color=#94a3b8"
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
        // The node invariant: one rounded-rect per state (the background adds a
        // second rect, so >= num_states).
        assert!(svg.matches("<rect").count() >= m.num_states as usize);
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