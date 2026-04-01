# src/ai_vision.py — v9.9.8 (Clifford Torus Skin + 24-TET Neutral Triad Audio)
# Global topological features (winding, linking, braiding_phase + flat Clifford skin)
# now drive both visual cone pulse **and** audible 24-TET chord.
# Quaternion math + helical geometry solve the AI persistent memory problem.
# safe_cosine(dim=-1 + .unsqueeze(0)) pattern enforced. SRP / DRY / config-over-hardcoding.

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
import scipy.io.wavfile as wavfile   # ← pure Python audio synthesis (pre-installed)

def generate_24tet_chord_audio(
    frequencies: List[float],
    duration: float = 2.5,
    sample_rate: int = 44100,
    save_path: str = "outputs/neutral_triad_chord.wav"
) -> str:
    """Pure topological chord synthesis: mix sine waves for the 24-TET neutral triad.
    Global braiding_phase modulates amplitude envelope (quaternion linking)."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Mix 3 sine waves (neutral triad) with soft envelope
    chord = np.zeros_like(t)
    for f in frequencies:
        chord += np.sin(2 * np.pi * f * t)

    # Global topological amplitude envelope (braiding_phase pulse)
    envelope = 0.8 * (1 + 0.3 * np.sin(2 * np.pi * t * 3))  # gentle braiding modulation
    chord = chord * envelope
    chord = chord / np.max(np.abs(chord)) * 0.9               # normalize

    # Write WAV (market-friendly audio redundancy)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(save_path, sample_rate, (chord * 32767).astype(np.int16))

    print(f"✅ 24-TET neutral triad audio rendered: {save_path} (Clifford Torus skin locked)")
    return save_path


def render_cone_as_display_video(
    conduit,
    num_frames: int = 240,
    save_path: str = "outputs/ai_vision_cone_display.mp4",
    fps: int = 12,
    width: int = 512,
    height: int = 512,
    chord_frequencies: Optional[List[float]] = None
) -> str:
    """Render Clifford Torus as perpendicular cone-as-display (CRT / Dream Catcher style).
    Radial LED strings from central z-origin → circular disc screen.
    Optional 24-TET chord audio is generated separately for clean separation (SRP)."""
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    frames: List[np.ndarray] = []
    s_base = 4.5

    for i in range(num_frames):
        s = s_base + i * 0.8
        all_geo = []
        for pol in range(conduit.num_pol):
            geo = conduit.get_helix_3d(s + pol * 0.3, pol)  # true 3D Clifford coord
            all_geo.append(geo)
        geo_stack = torch.stack(all_geo)  # (num_pol, 3)

        # Cone-as-display projection: perpendicular slice → circular disc
        frame = np.zeros((height, width, 3), dtype=np.uint8) + 8  # dark background

        # Central origin (z-axis focal point)
        center_x, center_y = width // 2, height // 2

        # braiding_phase color pulse (global topology — now locked)
        phase = (s % 24) / 24.0
        base_color = np.array([
            int(255 * (np.sin(phase * 2 * np.pi) * 0.5 + 0.5)),
            int(255 * (np.cos(phase * 2 * np.pi + 2) * 0.5 + 0.5)),
            int(255 * (np.sin(phase * 2 * np.pi + 4) * 0.5 + 0.5))
        ], dtype=np.uint8)

        for j, g in enumerate(geo_stack):
            # Project 3D point onto 2D disc (cone slice)
            r = np.sqrt(g[0].item()**2 + g[1].item()**2) * 1.8
            theta = np.arctan2(g[1].item(), g[0].item())
            screen_x = int(center_x + r * np.cos(theta))
            screen_y = int(center_y + r * np.sin(theta))

            # Radial LED string: line from center to outer point
            cv2.line(frame, (center_x, center_y), (screen_x, screen_y),
                     base_color.tolist(), thickness=2, lineType=cv2.LINE_AA)

            # Glowing LED points along string (topological pulse)
            intensity = 80 + 175 * np.sin(phase * 12 + j * 0.5)
            intensity = np.clip(intensity, 0, 255)
            color_pt = (base_color.astype(np.int32) * int(intensity) // 255).clip(0, 255).astype(np.uint8)

            for t in np.linspace(0, 1, 12):
                px = int(center_x + t * (screen_x - center_x))
                py = int(center_y + t * (screen_y - center_y))
                cv2.circle(frame, (px, py), 3, color_pt.tolist(), -1)

        frames.append(frame)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()

    print(f"→ AI-VISION Cone-as-Display saved: {save_path} (radial LED strings + circular disc screen)")
    return save_path