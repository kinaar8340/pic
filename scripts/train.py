# scripts/train.py — Persistent Identity Conduit Training Pipeline v2.0 (March 23, 2026)
# Supports optional VQC-Enhanced mode

import os
import sys
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import load_config, Config

# ──────────────────────────────────────────────────────────────────────
# Command-line arguments
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Persistent Identity Conduit")
parser.add_argument('--vqc', action='store_true', help='Use VQC-Enhanced Helical Conduit (OAM + Stevedore knot)')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
args = parser.parse_args()

USE_VQC = args.vqc

# Conditional import
if USE_VQC:
    from src.vqc_enhanced_conduit import VQCEnhancedHelicalConduit as ConduitClass
    print("🚀 Using VQC-Enhanced Helical Conduit (OAM flux + Stevedore topological protection)")
else:
    from src.conduit import TwistedHelicalConduit as ConduitClass
    print("Using standard TwistedHelicalConduit")

def resolve_device(device_str: str = "auto") -> str:
    if device_str.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str.lower()

def dummy_dataloader(
        embed_dim=384,
        num_samples=2048,  # total samples
        batch_size=64,
        n_clusters=6,
        points_per_cluster=None,  # if None → auto = num_samples // n_clusters
        depth_span_per_cluster=9.0,
        intra_cluster_noise=0.22,  # std of depth jitter within sequence
        drift_strength=0.014,
        noise_strength=0.052,
        norm_target=0.280,
):
    """
    Improved synthetic dataloader for PIC training.

    Changes:
    - Each cluster is now a mostly-sorted sequence along depth (increasing s)
    - Points within a cluster stay on the same polarization channel
    - Small random jitter added to depths → still mostly consecutive pairs exist
    - Clusters placed at staggered depth regions with some overlap margin
    - Final shuffle is very light or disabled to preserve more locality
      (you can control via shuffle_full argument below)
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

    if points_per_cluster is None:
        points_per_cluster = num_samples // n_clusters

    # Base directions for each cluster
    centers = torch.randn(n_clusters, embed_dim) * 0.38
    centers = centers / centers.norm(dim=1, keepdim=True)

    embs_list = []
    depths_list = []
    pols_list = []

    # Staggered starting points with partial overlap
    s_starts = torch.linspace(0.3, 11.5 - depth_span_per_cluster, n_clusters)

    for c in range(n_clusters):
        base = centers[c]

        # Drift vector — scale by the passed drift_strength
        drift_vec = torch.randn(embed_dim) * drift_strength

        # Sinusoidal non-linear component
        steps = torch.linspace(0, 1, points_per_cluster).unsqueeze(1)
        nonlinear_drift = torch.sin(steps * torch.pi * 2.2) * drift_vec * 1.5

        cluster_embs = base + nonlinear_drift
        cluster_embs += torch.randn_like(cluster_embs) * noise_strength * 1.4

        # Normalize to target norm
        cluster_embs = (
                cluster_embs
                / cluster_embs.norm(dim=1, keepdim=True).clamp_(min=1e-6)
                * norm_target
        )

        # Depth sequence
        s_base = torch.linspace(
            s_starts[c],
            s_starts[c] + depth_span_per_cluster,
            points_per_cluster
        )
        s_noise = torch.randn(points_per_cluster) * intra_cluster_noise
        s = s_base + s_noise
        s = torch.clamp(s, 0.05, 12.8)

        # Sort by depth → keeps consecutiveness
        sort_idx = torch.argsort(s)
        cluster_embs = cluster_embs[sort_idx]
        s = s[sort_idx]

        pol = torch.full((points_per_cluster,), c % 3, dtype=torch.long)

        embs_list.append(cluster_embs)
        depths_list.append(s)
        pols_list.append(pol)

    # Concatenate all clusters
    embs = torch.cat(embs_list)
    depths = torch.cat(depths_list)
    pols = torch.cat(pols_list)

    # Optional: very light shuffle or no shuffle at all
    # Goal: preserve most within-cluster consecutiveness
    # You can experiment with shuffle_full=True for more mixing
    shuffle_full = False
    if shuffle_full:
        perm = torch.randperm(len(embs))
        embs = embs[perm]
        depths = depths[perm]
        pols = pols[perm]
    else:
        # Optional: small block shuffle (keeps most sequences intact)
        block_size = 128
        n_blocks = len(embs) // block_size
        indices = torch.arange(len(embs))
        block_indices = indices[:n_blocks * block_size].view(n_blocks, block_size)
        block_perm = torch.randperm(n_blocks)
        shuffled_blocks = block_indices[block_perm].view(-1)
        # leftover tail stays in order
        tail = indices[n_blocks * block_size:]
        perm = torch.cat([shuffled_blocks, tail])
        embs = embs[perm]
        depths = depths[perm]
        pols = pols[perm]

    dataset = TensorDataset(embs, depths, pols)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=True,
        pin_memory=True, num_workers=6, persistent_workers=True
    )

@torch.no_grad()
def estimate_winding(conduit, pol_idx=0, n_points=200):
    s = torch.linspace(0.1, conduit.max_depth, n_points, device=conduit.device)
    pos = torch.stack([conduit.position(s_val.item(), pol_idx) for s_val in s])
    centered = pos - pos.mean(dim=0)
    angles = torch.atan2(centered[:, 1], centered[:, 0])
    delta_angles = torch.diff(angles)
    delta_angles = (delta_angles + math.pi) % (2 * math.pi) - math.pi
    total_wind = delta_angles.sum() / (2 * math.pi)
    return total_wind.item()

@torch.no_grad()
def plot_helix_pca(conduit, n_points=800, savepath="pca_helix.png"):
    conduit.eval()
    device = next(conduit.parameters()).device
    positions = []
    pol_labels = []
    s_vals = torch.linspace(0.1, conduit.max_depth, n_points, device=device)

    for pol in range(conduit.num_pol):
        pos = torch.stack([conduit.position(s.item(), pol) for s in s_vals]).cpu()
        positions.append(pos)
        pol_labels.append(torch.full((n_points,), pol, dtype=torch.long))

    all_pos = torch.cat(positions, dim=0).numpy()
    all_pol = torch.cat(pol_labels).numpy()

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(all_pos)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                         c=all_pol, cmap='tab10', alpha=0.75, s=18)
    ax.set_title(f"PCA projection of helical positions {'(VQC)' if USE_VQC else ''}")
    plt.colorbar(scatter, label="Polarization channel")
    plt.savefig(savepath, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"→ Saved PCA plot: {savepath}")

def main():
    torch.manual_seed(42)
    cfg: Config = load_config("configs/default.yaml")
    cfg.epochs = args.epochs

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Instantiate conduit
    if USE_VQC:
        conduit = ConduitClass(
            embed_dim=384,
            twist_rate=8.5,
            max_depth=48.0,
            num_polarizations=3,
            l_max=199,
            l_inner=1999,
            qec_level=16
        ).to(device)
        conduit.use_vqc_mode = True
    else:
        conduit = ConduitClass(
            embed_dim=384,
            twist_rate=8.5,
            max_depth=48.0,
            num_polarizations=3,
            quat_logical_dim=96
        ).to(device)

    optimizer = optim.AdamW(
        conduit.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    dataloader = dummy_dataloader(
        embed_dim=cfg.model.embed_dim,
        num_samples=cfg.data.num_samples,
        batch_size=cfg.data.batch_size,
        n_clusters=cfg.data.n_clusters,
        depth_span_per_cluster=cfg.data.depth_span_per_cluster,
        intra_cluster_noise=cfg.data.intra_cluster_noise,
        drift_strength=cfg.data.drift_strength,
        noise_strength=cfg.data.noise_strength,
    )

    print(f"Starting training loop — {'VQC mode' if USE_VQC else 'Standard mode'} ...")

    best_recall_cos = -1.0

    for epoch in range(cfg.epochs):
        conduit.train()
        total_batch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch_emb, batch_s, batch_pol = [t.to(device) for t in batch]

            inputs = [
                {
                    'emb': batch_emb[i],
                    's': float(batch_s[i].item()),
                    'pol_idx': int(batch_pol[i].item()),
                }
                for i in range(batch_emb.size(0))
            ]

            metrics = conduit.training_step(
                inputs=inputs,
                optimizer=optimizer,
                recon_weight=cfg.training.recon_weight,
                mag_weight=cfg.training.mag_weight,
                align_weight=getattr(cfg.training, 'align_weight', 520.0),
                locality_weight=cfg.training.locality_weight,
                braiding_weight=getattr(cfg.training, 'braiding_weight', 8.0),
                winding_weight=getattr(cfg.training, 'winding_weight', 4.5),
            )

            total_batch_loss += metrics['total']
            num_batches += 1

        avg_epoch_loss = total_batch_loss / num_batches if num_batches > 0 else 0.0

        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            print(f"Epoch {epoch:4d} | total={avg_epoch_loss:8.5f} | "
                  f"recon={metrics.get('recon',0):8.5f} | align={metrics.get('align',0):8.5f} | "
                  f"locality={metrics.get('locality',0):8.5f}")

        # Evaluation & visualization
        if (epoch % cfg.training.eval_every == 0 or epoch == cfg.epochs - 1) and not args.no_viz:
            plot_helix_pca(conduit, savepath=f"pca_helix_epoch_{epoch:04d}.png")
            stats = conduit.monitor_topological_winding()
            print(f"✓ Geometric: {stats['geometric_winding']:.3f} | "
                  f"Effective: {stats['effective_winding']:.3f} | "
                  f"Learned: {stats['learned_contribution']:.3f}")
            if USE_VQC:
                print(f"  VQC → OAM L_max={getattr(conduit,'l_max','N/A')} | "
                      f"Inner winding={stats.get('inner_vortex_winding',0):.4f}")

    # Final evaluation
    print("\n" + "═" * 70)
    print("Training finished. Final evaluation:")

    conduit.eval()
    with torch.no_grad():
        cos_sims = []
        sample_inputs = []
        max_samples = 200

        for sample_batch in dataloader:
            s_emb, s_s, s_pol = [t.to(device) for t in sample_batch]
            for i in range(s_emb.size(0)):
                if len(sample_inputs) >= max_samples:
                    break
                sample_inputs.append({
                    'emb': s_emb[i],
                    's': float(s_s[i].item()),
                    'pol_idx': int(s_pol[i].item()),
                })
            if len(sample_inputs) >= max_samples:
                break

        for item in sample_inputs:
            recalled = conduit.read(item['s'], item['pol_idx'])
            cos = F.cosine_similarity(recalled.view(1, -1), item['emb'].view(1, -1)).item()
            cos_sims.append(cos)

        cos_arr = np.array(cos_sims)
        print(f"  Mean recall cosine    : {cos_arr.mean():.4f} ± {cos_arr.std():.4f}")
        print(f"  Cosine > 0.80         : {100 * (cos_arr > 0.80).mean():5.1f}%")

    # Final visualizations
    if not args.no_viz:
        plot_helix_pca(conduit, savepath="pca_helix_final.png")
        conduit.render_braided_lattice_style(save_path="pic_braided_lattice_final.png")
        conduit.render_microtubule_style(save_path="pic_microtubule_final.png")

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()