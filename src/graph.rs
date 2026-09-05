//! The CUDA graph host-side interface (the PdaGraph).
//!
//! A CUDA graph is a captured DAG of kernel launches + memcpys, replayed with
//! per-replay param updates and no re-capture. This module defines the graph
//! structure (the nodes + the edges + the static/dynamic tiers) + a CPU mock
//! replay that proves the DAG semantics (the topological execution). The
//! attention-rs implements the actual CUDA capture/replay against this same
//! structure (the layout-identity invariant).
//!
//! The compactness comes from the research advantages:
//!   - Pre3 prefix-conditioned edges -> the static DAG (the no runtime branch).
//!   - PSC codebook -> the small mask global (the few mask-classes).
//!   - SWYB d_H bound -> the fixed K+1 unroll (the bounded graph size).
//!   - the static bitvec -> the graph body is only the kernel nodes + the
//!     state memcpys (the table is a global, not a node node).

use crate::machine::PdaMachine;
use crate::pda::PdaStream;

/// A graph node (the a kernel launch OR a memcpy edge).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphNode {
    /// The pda_compute_masks kernel (the per-seq mask, the batch node 1 (mask)).
    ComputeMasks { batch: usize },
    /// The pda_sample kernel (the greedy/top-k/top-p with the mask).
    Sample { batch: usize },
    /// The pda_advance kernel (the per-seq step, the pushdown update).
    Advance { batch: usize },
    /// The pda_project_masks kernel (the K+1 masks, the drafting).
    ProjectMasks { batch: usize, k: usize },
    /// A H2D memcpy (the per-seq state upload).
    H2D { bytes: usize },
    /// A D2H memcpy (the next-state download).
    D2H { bytes: usize },
}

/// The CUDA graph (the captured DAG).
#[derive(Debug, Clone)]
pub struct PdaGraph {
    /// The static tier (the graph globals, the uploaded once).
    pub bitvec_len: usize,
    pub codebook_len: usize,
    /// The nodes (the kernel launches + the memcpys).
    pub nodes: Vec<GraphNode>,
    /// The edges (the (src, dst) node indices, the DAG).
    pub edges: Vec<(usize, usize)>,
}

impl PdaGraph {
    /// Build the single-token decode graph (the 3 kernels + the state memcpys).
    /// The DAG: H2D -> ComputeMasks -> Sample -> Advance -> D2H.
    pub fn single_token_decode(batch: usize) -> Self {
        let n_h2d = 0;
        let n_masks = 1;
        let n_sample = 2;
        let n_advance = 3;
        let n_d2h = 4;
        PdaGraph {
            bitvec_len: 0, // the filled at the upload
            codebook_len: 0,
            nodes: vec![
                GraphNode::H2D { bytes: batch * 12 }, // the ctrl + the stack[8] + the sp
                GraphNode::ComputeMasks { batch },
                GraphNode::Sample { batch },
                GraphNode::Advance { batch },
                GraphNode::D2H { bytes: batch * 12 },
            ],
            edges: vec![
                (n_h2d, n_masks),
                (n_masks, n_sample),
                (n_sample, n_advance),
                (n_advance, n_d2h),
            ],
        }
    }

    /// Build the drafting graph (the K+1 projection, the fixed unroll). The DAG:
    /// H2D -> ProjectMasks(K+1) -> D2H. The K is fixed at capture (the SWYB d_H
    /// bound), so the graph size is bounded.
    pub fn drafting(batch: usize, k: usize) -> Self {
        let n_h2d = 0;
        let n_project = 1;
        let n_d2h = 2;
        PdaGraph {
            bitvec_len: 0,
            codebook_len: 0,
            nodes: vec![
                GraphNode::H2D { bytes: batch * 12 },
                GraphNode::ProjectMasks { batch, k },
                GraphNode::D2H { bytes: batch * (k + 1) * 4 },
            ],
            edges: vec![(n_h2d, n_project), (n_project, n_d2h)],
        }
    }

    /// The topological order of the nodes (the execution order). Returns
    /// None if the graph has a cycle (the invalid DAG).
    pub fn topo_order(&self) -> Option<Vec<usize>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(s, d) in &self.edges {
            if s >= n || d >= n {
                return None;
            }
            adj[s].push(d);
            in_degree[d] += 1;
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::new();
        while let Some(&u) = queue.first() {
            queue.remove(0);
            order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }
        if order.len() == n {
            Some(order)
        } else {
            None // the cycle
        }
    }

    /// The CPU mock replay: execute the nodes in topological order (the
    /// simulate the CUDA graph replay on the CPU). This proves the DAG
    /// semantics (the topological execution) without a GPU.
    pub fn mock_replay(&self, machine: &PdaMachine) -> Result<Vec<u32>, GraphError> {
        let order = self.topo_order().ok_or(GraphError::Cycle)?;
        let mut state: Vec<u32> = vec![machine.start_state; 1];
        for &node_idx in &order {
            match &self.nodes[node_idx] {
                GraphNode::ComputeMasks { batch } => {
                    // the batched mask (the batch node 1 (mask))
                    let configs: Vec<(u32, Vec<u32>)> =
                        (0..*batch).map(|_| (state[0], vec![machine.start_stack])).collect();
                    let _ = machine.mask_batch(&configs);
                }
                GraphNode::Sample { batch } => {
                    // the sample (the greedy, the first legal input)
                    let _ = batch;
                }
                GraphNode::Advance { batch } => {
                    // the advance (the batch node 2 (step))
                    let batch2: Vec<((u32, Vec<u32>), u32)> = (0..*batch)
                        .map(|_| ((state[0], vec![machine.start_stack]), 0u32))
                        .collect();
                    let stepped = machine.step_batch(&batch2);
                    if let Some((q, _)) = stepped.first() {
                        state = vec![*q];
                    }
                }
                GraphNode::ProjectMasks { batch, k } => {
                    // the projection (the batch node 3 (project), the K+1 masks)
                    let configs: Vec<(u32, Vec<u32>)> = (0..*batch)
                        .map(|_| (state[0], vec![machine.start_stack]))
                        .collect();
                    let drafts: Vec<Vec<u32>> = vec![vec![0u32; *k]; *batch];
                    let _ = machine.project_batch(&configs, &drafts);
                }
                GraphNode::H2D { .. } | GraphNode::D2H { .. } => {
                    // the memcpys are no-ops in the mock (the data is in-memory)
                }
            }
        }
        Ok(state)
    }
}

/// Errors from the graph replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The graph has a cycle (the invalid DAG).
    Cycle,
}
impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Cycle => write!(f, "the graph has a cycle (the invalid DAG)"),
        }
    }
}
impl std::error::Error for GraphError {}