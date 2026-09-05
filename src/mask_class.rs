//! The PSC mask-classification (arXiv:2608.03065).
//!
//! The parser-stack classification: the config -> the mask-class (the code) ->
//! the mask (the VOB). The codebook is the set of distinct masks. Many configs
//! share the same mask (the catch-all configs share the universal mask),
//! so the codebook is much smaller than the config space.
//!
//! This is the PSC's core idea: the mask is a CODEBOOK lookup (the
//! config -> the code -> the VOB), not a per-config computation.

use std::collections::HashMap;

use crate::machine::PdaMachine;

/// The mask-class (the code, the u32).
pub type MaskClass = u32;

/// The codebook (the mask-class -> the mask, the VOB as a sorted token<u32>).).
#[derive(Debug, Clone, Default)]
pub struct Codebook {
    // the mask-class -> the allowed tokens (the sorted Vec)32>)
    pub classes: HashMap<MaskClass, Vec<u32>>,
    // the next free mask-class ID
    next_class: MaskClass,
}

impl Codebook {
    pub fn new() -> Self {
        Codebook {
            classes: HashMap::new(),
            next_class: 0,
        }
    }

    /// Intern a mask (the allowed tokens) into the codebook. Returns the
    /// mask-class (the code). The the same mask (the same allowed set)
    /// always gets the same code (the dedup).
    pub fn intern(&mut self, allowed: &[u32]) -> MaskClass {
        // the dedup: the same allowed set -> the same code
        for (&code, existing) in self.classes.iter() {
            if existing == allowed {
                return code;
            }
        }
        let code = self.next_class;
        self.next_class += 1;
        self.classes.insert(code, allowed.to_vec());
        code
    }

    /// The the codebook lookup (the mask-class -> the allowed tokens).
    pub fn mask(&self, code: MaskClass) -> Option<&Vec<u32>> {
        self.classes.get(&code)
    }

    /// The the number of distinct masks (the codebook size).
    pub fn len(&self) -> usize {
        self.classes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }
}

/// The the PSC classifier (the config -> the mask-class). The the classifier
/// maps each (state, stack-top) config to its mask-class (the code).
#[derive(Debug, Clone)]
pub struct PscClassifier {
    // the config (the state + the stack-top) -> the mask-class
    pub config_to_class: HashMap<(u32, u32), MaskClass>,
    pub codebook: Codebook,
}

impl PscClassifier {
    /// Build the PSC classifier from the machine (the config -> the mask-class).
    /// The the mask at a config is the set of inputs with a defined transition.
    pub fn build(machine: &PdaMachine) -> Self {
        let mut classifier = PscClassifier {
            config_to_class: HashMap::new(),
            codebook: Codebook::new(),
        };
        for state in 0..machine.num_states {
            for top in 0..machine.num_stack_syms {
                // the mask at (state, top) = the inputs with a defined transition
                let allowed: Vec<u32> = (0..=machine.num_inputs)
                    .filter(|&a| !machine.lookup(state, Some(a), top).is_empty())
                    .collect();
                if allowed.is_empty() {
                    continue;
                }
                let code = classifier.codebook.intern(&allowed);
                classifier.config_to_class.insert((state, top), code);
            }
        }
        classifier
    }

    /// The the mask-class for a config (the state + the stack-top).
    pub fn class_of(&self, state: u32, top: u32) -> Option<MaskClass> {
        self.config_to_class.get(&(state, top)).copied()
    }

    /// The the mask for a config (the mask-class -> the allowed tokens).
    pub fn mask_of(&self, state: u32, top: u32) -> Option<Vec<u32>> {
        let code = self.class_of(state, top)?;
        self.codebook.mask(code).cloned()
    }
}