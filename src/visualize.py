# src/visualize.py

import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Optional


@torch.no_grad()
def visualize_helix(
        self,
        n_points: int = 800,
        pol_colors: List[str] = ['#1f77b4', '#ff7f0e', '#2ca02c'],
        title: str = "Twisted Helical Manifold (Quaternion-enhanced)",
        save_path: Optional[str] = None,
):
    """Plot 3D projection of the helical manifold positions across polarizations."""

    # ... (your existing plotting code) ...
    # fig = plt.figure(...)
    # ax = fig.add_subplot(...)
    # ... plotting happens here ...

    # ────────────────────────────────────────────────
    # Handle saving
    if save_path is not None:
        # User gave explicit path → use it directly
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
    else:
        # Auto-save with timestamp / hash / version / whatever you prefer
        output_dir = Path.home() / "pic" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Example filename patterns — choose one:

        # Option A: timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"helix_manifold_{timestamp}.png"

        # Option B: simple counter / version
        # filename = f"helix_manifold_v{self.num_branches or 1}.png"

        # Option C: include twist rate & pol count
        # filename = f"helix_t{self.twist_rate:.1f}_p{self.num_pol}_{n_points}pts.png"

        save_path = output_dir / filename

        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    plt.show()
    # plt.close(fig)  # optional — prevents memory buildup in loops

    if __name__ == "__main__":
        conduit = TwistedHelicalConduit(embed_dim=384, quat_logical_dim=96)
        conduit.visualize_helix(n_points=1200, save_path="helix_manifold.png")