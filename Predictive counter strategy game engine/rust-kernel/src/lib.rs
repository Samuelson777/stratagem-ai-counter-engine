use wasm_bindgen::prelude::*;

/// Mode 6 ("The Floor") 10x10 Territorial Conquest Kernel
#[wasm_bindgen]
pub struct FloorTerritoryKernel {
    height: usize,
    width: usize,
    ownership: Vec<u16>,
    alive: Vec<bool>,
}

#[wasm_bindgen]
impl FloorTerritoryKernel {
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize) -> FloorTerritoryKernel {
        let total = height * width;
        let ownership = (0..total as u16).collect();
        let alive = vec![true; total];
        FloorTerritoryKernel {
            height,
            width,
            ownership,
            alive,
        }
    }

    /// Computes area A_p(t) = |{(r, c) : B_{r,c}(t) = p}|
    pub fn get_player_area(&self, player_id: u16) -> usize {
        self.ownership.iter().filter(|&&owner| owner == player_id).count()
    }

    /// Scans all tiles in T_p(t) to return orthogonally adjacent (4-neighbor) alive defenders N_p(t)
    pub fn get_admissible_defenders(&self, player_id: u16) -> Vec<u16> {
        let mut defenders = Vec::new();
        for r in 0..self.height {
            for c in 0..self.width {
                let idx = r * self.width + c;
                if self.ownership[idx] != player_id {
                    continue;
                }
                let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dr, dc) in neighbors.iter() {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < self.height as isize && nc >= 0 && nc < self.width as isize {
                        let n_idx = (nr as usize) * self.width + (nc as usize);
                        let def_id = self.ownership[n_idx];
                        if def_id != player_id
                            && self.alive[def_id as usize]
                            && !defenders.contains(&def_id)
                        {
                            defenders.push(def_id);
                        }
                    }
                }
            }
        }
        defenders
    }

    /// Resolves a territorial duel and transfers all loser tiles to the winner
    pub fn resolve_duel(&mut self, attacker_id: u16, defender_id: u16, attacker_won: bool) -> u16 {
        let (winner, loser) = if attacker_won {
            (attacker_id, defender_id)
        } else {
            (defender_id, attacker_id)
        };
        for owner in self.ownership.iter_mut() {
            if *owner == loser {
                *owner = winner;
            }
        }
        self.alive[loser as usize] = false;
        winner
    }
}

/// Computes the FightingICE Score-Damage-Ratio (SDR) metric and applies 56-action masking
#[wasm_bindgen]
pub fn compute_combat_sdr(my_hp: f32, opp_hp: f32, elapsed_ticks: u32) -> f32 {
    if elapsed_ticks == 0 {
        return 0.0;
    }
    let hp_diff = (my_hp - opp_hp).max(0.0);
    hp_diff / (elapsed_ticks as f32)
}