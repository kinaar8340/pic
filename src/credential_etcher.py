# src/credential_etcher.py — v9.9.1 (Clifford Torus Skin + Toroidal 3-6-9 Knots)
# Global topological invariants drive primary persistence.
# JSON fallback is market redundancy only.
# SRP, DRY, Dependency Inversion, Configuration over Hardcoding, safe_cosine enforced.

import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet

from src.conduit import TwistedHelicalConduit, safe_cosine
from src.config import load_config


@dataclass
class UserIntakeForm:
    """Explicit schema — configuration over hardcoding."""
    username: str
    public_email: str
    x_handle: str = "@kinaar8340"
    private_email: Optional[str] = None
    webapp_credentials: Dict[str, Dict[str, str]] = None

    def to_topological_embedding(self, embedder) -> torch.Tensor:
        """One canonical string → helical position (global topology first)."""
        text = (
            f"User intake: {self.username} | public:{self.public_email} | x:{self.x_handle} | "
            f"private:{self.private_email or '—'} | creds:{len(self.webapp_credentials or {})}"
        )
        emb = embedder.encode(text, convert_to_tensor=True)
        return F.normalize(emb, dim=-1) * 0.28


class CredentialEtcher:
    """SRP: intake → topological bake + encrypted JSON fallback.
    Depends only on abstract conduit + config (Dependency Inversion)."""

    def __init__(self, conduit: TwistedHelicalConduit, embedder, device: str = "cuda"):
        self.conduit = conduit
        self.embedder = embedder
        self.device = device
        self.cfg = load_config()                                      # ← fully config-driven

        # Paths now come from configs/default.yaml (no hardcoding)
        self.fernet_key_path = Path(self.cfg.credentials.fernet_key_path)
        self.json_path       = Path(self.cfg.credentials.json_fallback_path)
        self.default_pol_idx = self.cfg.credentials.default_pol_idx
        self.default_s_start = self.cfg.credentials.default_s_start

        self._ensure_key()                                            # ← now guaranteed

    def _ensure_key(self):
        """One-time symmetric key (never committed)."""
        self.fernet_key_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.fernet_key_path.exists():
            self.fernet_key_path.write_bytes(Fernet.generate_key())
        self.fernet = Fernet(self.fernet_key_path.read_bytes())      # ← attribute now always exists

    def etch(self, form: UserIntakeForm, s_start: Optional[float] = None) -> Dict[str, Any]:
        """Primary: bake into global topology (Clifford Torus skin + braiding_phase).
        Fallback: encrypted JSON (local vector state for third-party apps)."""
        emb = form.to_topological_embedding(self.embedder).to(self.device)

        # ── Primary topological etch (global invariants locked) ──
        item = {"emb": emb, "s": s_start or self.default_s_start, "pol_idx": self.default_pol_idx}
        self.conduit.training_step(inputs=[item], optimizer=None)      # pure bake
        self.conduit.bake_to_forked_cube(
            cube_idx=11,
            emb=emb,
            orientation=24,
            parent_idx=0
        )

        stats = self.conduit.monitor_topological_winding()

        # ── Market-friendly JSON fallback ──
        payload = asdict(form)
        payload["topological_fingerprint"] = {
            "braiding_phase": float(stats.get("braiding_phase", 0.0)),
            "effective_winding": float(stats.get("effective_winding", 0.0)),
            "geometric_winding": float(stats.get("geometric_winding", 0.0)),
            "clifford_projection": bool(stats.get("clifford_projection", False)),
        }
        encrypted = self.fernet.encrypt(json.dumps(payload, ensure_ascii=False).encode())
        self.json_path.write_bytes(encrypted)

        print(f"✅ Etched credentials → braiding_phase={stats.get('braiding_phase',0):.6f} | JSON fallback written")

        return {
            "status": "etched",
            "braiding_phase": stats.get("braiding_phase", 0.0),
            "json_fallback_path": str(self.json_path),
            "recall_cos": safe_cosine(                                 # enforced pattern
                emb.unsqueeze(0), self.conduit.read(s_start or self.default_s_start, pol_idx=self.default_pol_idx).unsqueeze(0)
            ).item()
        }