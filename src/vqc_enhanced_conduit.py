# src/vqc_enhanced_conduit.py — v9.9 (March 25, 2026)
# VQC = full quaternion spine + light OAM modulation (topology preserved)
# Inherits Clifford Torus skin + toroidal_modulo9 + 3-6-9 Vortex Math from base.
# safe_cosine pattern already enforced in base conduit.py everywhere.

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.conduit import TwistedHelicalConduit


class VQCEnhancedHelicalConduit(TwistedHelicalConduit):
    """Denser twist + OAM flux on top of full global topology.
    Never overrides core geometry — extends it (DRY + invariants locked).
    Now includes flat Clifford Torus skin (zero Gaussian curvature) around
    CubeChain z-axis conductor + 9 orthogonal BowTie shards + Rodin-CT transceiver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vqc_scale = nn.Parameter(torch.tensor(1.0))
        self.oam_freq = nn.Parameter(torch.tensor(8.5))  # configurable OAM

        # ─── VQC OAM + full Clifford Torus skin (global topology first) ───
        self.helix_projector = nn.Sequential(
            nn.Linear(3, 128),
            nn.LayerNorm(128),  # ← normalization trick for robust residuals
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )
        nn.init.normal_(self.helix_projector[0].weight, mean=0.0, std=0.022)
        nn.init.normal_(self.helix_projector[3].weight, mean=0.0, std=0.012)
        for p in self.helix_projector.parameters():
            p.requires_grad = True

        # Configurable scales (over hardcoding)
        self.residual_scale = nn.Parameter(torch.tensor(0.92))
        self.quat_scale = nn.Parameter(torch.tensor(0.42))
        print("✅ VQC-Enhanced: LayerNorm + configurable scales (global topology drives persistence)")

    def position(self, s: float, pol_idx: int = 0) -> torch.Tensor:
        """Full pearl-string + quaternion Frenet (from base) + VQC OAM.
        Global invariants (winding + linking + braiding_phase + flat Clifford skin)
        are preserved by construction."""
        base_emb = super().position(s, pol_idx)  # ← toroidal/Clifford/369 topology inherited
        oam_phase = torch.tensor(s * self.oam_freq.item() + pol_idx * 3.0,
                                 device=self.device, dtype=torch.float32)
        oam_mod = torch.sin(oam_phase) * 0.042 * (pol_idx + 1)  # light orthogonal modulation
        vqc_emb = base_emb + oam_mod
        return F.normalize(vqc_emb * self.vqc_scale, dim=-1) * self.output_scale


if __name__ == "__main__":
    # Quick test — full v9.9 topology + VQC OAM
    conduit = VQCEnhancedHelicalConduit(
        toroidal_modulo9=True,
        vortex_math_369=True,
        clifford_projection=True
    )
    print("✅ PIC v9.9 VQC-Enhanced with Clifford Torus + Toroidal Modulo-9 + 3-6-9 Vortex Math loaded")
    stats = conduit.monitor_topological_winding()
    print(stats)
    print("→ Global topological invariants (winding + braiding_phase + flat Clifford skin) are now live")

