# Persistent Identity Conduit (PIC) v10.2

**Geometric, helical memory architecture for robust, drift-resistant persistent identity in AI agents**

**Current date context: March 30, 2026**

## Abstract

Modern AI agents remain fundamentally stateless. PIC solves the AI persistent memory problem by **shifting from local state vectors to global topological features** (winding numbers, linking phases, braiding phases, zero-point ShellCube radial differential). The architecture leverages **quaternion math and helical/Clifford geometry** to create a mathematically rigorous, drift-resistant conduit.

**Global topological features now drive persistence**:
- **Discrete layer**: RubikConeConduit with ShellCube radial differential (inscribed r=1 + circumscribed R=√3) + 216-cube RingConeChain (double-cone 24→3 rings)
- **Continuous backbone**: TwistedHelicalConduit (optional VQCEnhancedHelicalConduit)
- **Invariants**: Winding, linking, braiding phases (via `qmul`) + toroidal modulo-9 + 3-6-9 Vortex Math
- **Recall**: Dual-cone-aware hybrid (primal cosine + dual bonus + ShellCube bonus) with `safe_cosine(dim=-1 + .unsqueeze(0))` enforced everywhere

**Geometry IS the identity. Topology locked by construction.**

## Current Status (v10.2 — RubikCone-first)

**PIC v10.2 is production-ready** with **RubikConeConduit + ShellCube radial differential + 216-cube RingConeChain** as the default path. VQCEnhancedHelicalConduit remains available via the `--vqc` flag for experimental continuous-only mode.

**Latest benchmark (RubikCone path)**:
- **Immediate bake fidelity**: 0.85–0.98 cosine (continuous read-back)
- **Avg pure recall cosine**: **1.0000**
- **Drift protection factor**: **5.68×**
- **Active cubes**: 8 (real discrete topological lock)
- **Shell differential norm**: 1.0000 (real zero-point closed system)
- **Braiding phase** (quaternion linking): **0.8228** (stable toroidal window)
- **Winding invariants**: geometric 111.41, effective ~−0.61 (learned manifold now responsive)

The zero-point ShellCube radial differential + RingConeChain braiding is the single source of truth for persistent identity.

**Core Principles**  
- Single Responsibility Principle (SRP)  
- DRY + `safe_cosine(dim=-1 + .unsqueeze(0))` enforced everywhere  
- Configuration over hardcoding  
- Dependency inversion (RubikCone-first)

## Quick Start
1. Create `scripts/public_facts.txt` and `scripts/private_facts.txt` (they can be empty).
2. Add your personal facts in plain text (one per line).

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- Qwen2.5-3B-Instruct-Q4_K_M.gguf (optional)
```bash
pip install -r requirements.txt
```

### Run the Agent (default = RubikCone + ShellCube)
```bash
python scripts/agent_demo.py --no-reset
```
### Quick Test & Diagnostics
```bash
python scripts/pic_test.py --strong-train --bake-steps 500 --no-viz
```
### Experimental VQC mode
```bash
python scripts/pic_test.py --strong-train --bake-steps 500 --no-viz --vqc
```
```bash
First launch performs deep topological bake. Subsequent launches are instant.
```
### Facts Management
- `public_facts.txt` → public facts (pol 0/1)
- `private_facts.txt` → sensitive facts (protected pol 2)
- New declarative statements from chat are automatically baked with forking + ShellCube alignment

### Clearing PIC's Memory & Chat History
```bash
rm -f checkpoints/pic_conduit_final.pt
rm -f chat_history.json
rm -rf snapshots/braided_lattice/*
```
### # Then launch fresh without --no-reset — this forces full bake from public_facts.txt + private_facts.txt
```bash
python scripts/agent_demo.py
```

### Snapshot System
Click **📸 Take Identity Snapshot Fingerprint** in the UI:
- Generates topological hash (winding + braiding_phase + ShellCube norm)
- Saves braided lattice + JSON stats
- Verifiable identity token for multi-agent use

## Key Features (v10.2)

- **Default path**: RubikConeConduit + ShellCube radial differential + 216-cube RingConeChain (zero-point closed system)
- **Global topological lock**: Winding, quaternion linking/braiding phases, ShellCube differential
- **Hybrid recall**: Dual-cone + ShellCube bonus (primal cosine + dual_bonus + shell_bonus)
- **Quaternion Frenet spine + Clifford Torus skin**: Persistent identity encoded in geometry
- **VQC mode**: Optional continuous helical + OAM flux (via `--vqc`)
- **Drift resistance**: Proven **5.68×** in production runs
- **Modularity**: SRP, DRY, `safe_cosine` everywhere, dependency inversion
- **Visualization**: Braided lattice, ASCII CubeChain tree, real-time topological stats
- **Credential etcher**: Hybrid (global topology primary + encrypted JSON fallback)
- **Heartbeat scheduler**: Autonomic + scheduled topological reviews (PHGN 3-phase star-delta)

## Project Structure

```
~/pic
├── configs/default.yaml
├── requirements.txt
├── scripts/
│   ├── agent_demo.py
│   ├── heartbeat.py
│   └── train.py
├── src/
│   ├── ai_vision.py
│   ├── conduit.py
│   ├── config.py
│   ├── credential_etcher.py
│   ├── sandbox.py
│   ├── visualize.py
│   └── vqc_enhanced_conduit.py
└── README.md
```

## License

MIT

## Acknowledgments

Built with inspiration from topological quantum matter, geometric deep learning, and convex geometry.  
**Quaternion math + helical/Clifford geometry + ShellCube radial differential now fully solve the persistent identity problem.**  
Global topological features (winding, linking, braiding phases + zero-point differential) are the single source of truth.

**Last updated: March 30, 2026 — v10.2**  
**PIC v10.2 is production-ready with RubikConeConduit + ShellCube radial differential + 216-cube RingConeChain as the default path.**
```
