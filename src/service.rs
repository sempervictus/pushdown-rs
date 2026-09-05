//! The SIMD service (the device model). The scalar code is the HOST; this
//! service is the DEVICE (the SIMD, the mock of the GPU). The host sends
//! a BATCH (the (config, token) pairs); the device processes the batch in
//! parallel (the SIMD lanes); the host receives the result (the
//! (next-config, mask) pairs). The host never sees the SIMD - it just sends/
//! receives batches (the batched pipeline node boundary, the GPU host-device model).

use crate::machine::PdaMachine;
use crate::pda::PdaStream;

/// The SIMD service (the device). The host sends a batch; the device
/// processes it in parallel; the host receives the result.
pub struct PdaService {
    machine: PdaMachine,
}

impl PdaService {
    pub fn new(machine: PdaMachine) -> Self {
        PdaService { machine }
    }

    /// The device's step (the batch node 2 (step)): the batch of (config, token) ->
    /// the batch of next-config. The host sends the batch; the device processes
    /// it in parallel (the SIMD lanes); the host receives the result.
    pub fn step(&self, batch: &[(u32, Vec<u32>, u32)]) -> Vec<(u32, Vec<u32>)> {
        let configs: Vec<(u32, Vec<u32>)> = batch.iter().map(|(q, s, _)| (*q, s.clone())).collect();
        let tokens: Vec<u32> = batch.iter().map(|(_, _, t)| *t).collect();
        let pairs: Vec<((u32, Vec<u32>), u32)> =
            configs.iter().zip(tokens.iter()).map(|(c, t)| (c.clone(), *t)).collect();
        self.machine.step_batch(&pairs)
    }

    /// The device's mask (the batch node 1 (mask)): the batch of config -> the batch
    /// of mask. The host sends the batch; the device processes it in parallel.
    pub fn mask(&self, configs: &[(u32, Vec<u32>)]) -> Vec<Vec<u32>> {
        self.machine.mask_batch(configs)
    }

    /// The device's projection (the batch node 3 (project)): the batch of (config,
    /// draft[K]) -> the batch of (K+1 masks). The drafting use-case.
    pub fn project(
        &self,
        configs: &[(u32, Vec<u32>)],
        drafts: &[Vec<u32>],
    ) -> Vec<Vec<Vec<u32>>> {
        self.machine.project_batch(configs, drafts)
    }

    /// The device's machine (the read-only access, the inspection).
    pub fn machine(&self) -> &PdaMachine {
        &self.machine
    }
}