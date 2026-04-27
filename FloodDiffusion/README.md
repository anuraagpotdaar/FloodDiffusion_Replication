---
title: FloodDiffusion Motion Demo
emoji: "\U0001F30A"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hardware: t4-small
pinned: false
license: apache-2.0
short_description: "Streaming motion demo, rigged mesh, pose-aware control"
---

# FloodDiffusion Replication — Streaming Motion Demo

A replication / extension of the **FloodDiffusion** streaming motion-generation demo
([CVPR 2026 Highlight, Cai *et al.*](https://arxiv.org/abs/2512.03520)). Original
authors: Yiyi Cai, Yuhan Wu, Kunhang Li, You Zhou, Bo Zheng,
[Haiyang Liu](https://h-liu1997.github.io/). This Space replicates and extends the
authors' Hugging Face streaming demo
([H-Liu1997/FloodDiffusion-Streaming](https://huggingface.co/spaces/H-Liu1997/FloodDiffusion-Streaming))
on top of the [`ShandaAI/FloodDiffusionTiny`](https://huggingface.co/ShandaAI/FloodDiffusionTiny)
model.

## What this fork adds

The upstream demo renders motion as a colored stick figure and exposes a single text
input. This fork replaces the renderer with a rigged humanoid, adds a pose-aware
control loop, and tightens the streaming behaviour for low-latency interaction.

### Rendering

- **Rigged Soldier mesh** instead of stick figure (`static/js/mesh_body.js`,
  `static/models/Soldier.glb`). The 22 HumanML3D joint positions retarget onto the
  rig each frame using parent-local quaternion math; a yaw-only frame correction on
  the Hips bone recovers the rotation that aim-at-child loses on near-vertical
  bones.
- **Standing pose on load** — the rig snaps to a hardcoded standing pose
  (`MeshBody.STANDING_POSE`) so the character never appears in T-pose between page
  load and the first generated frame.
- **Full-window canvas with floating top-right control panel**.

### Control loop

- **Chip-based motion picker** — 14 preset actions (Walk, Run, Jump, Dance, Sit,
  Stand, Wave, Kick, Punch, Crouch, Turn, Clap, Bow, Circle). Each chip is
  re-clickable (bypasses dedupe) and auto-deselects after a short flash.
- **Auto-fire on space / Enter / typing pause** — typing into the prompt box
  pushes the new prompt as you compose, with a 600 ms typing-pause debounce as the
  fallback.
- **Pose-aware transitions** — `model_manager.py` carries a small motion catalog
  tagging each prompt with `type` (`continuous` / `oneshot`), `ends_in`
  (`standing` / `sitting` / `crouching`), and `needs_starting`. If the requested
  motion needs the character standing but the character is sitting (or
  crouching), the system queues the requested motion behind a "stand up"
  transition, then plays the original motion.
- **Continuous vs one-shot** — Walk/Run/Dance/Turn/Circle loop until you click
  another chip; Jump/Sit/Stand/Wave/Kick/Punch/Crouch/Clap/Bow play once and end
  in their natural pose.

### Streaming behaviour

- **Instant text response** — `update_text` resets the model's autoregressive
  latent state (`init_generated`) so the next chunk is fully conditioned on the
  new prompt instead of bleeding the old motion's dynamics. Without this, the
  model continues the previous trajectory regardless of the new prompt.
- **Warmup-discard** — the first chunk after every latent reset (5 frames) comes
  from random noise and looks glitchy ("character falls" on run→stand). Those
  frames are dropped silently before reaching the buffer; the visible motion only
  begins once the model has converged on the new prompt.
- **Settle detection** — one-shot motions stop when their per-frame joint
  displacement has been below threshold for a 16-frame window (after a 80-frame
  minimum). Continuous motions never auto-end.
- **Smaller buffers** — server target buffer 4 frames, client batch 2 frames
  (down from upstream's 16/8) to reduce input-to-screen latency on T4 ≤ 200 ms.

### CPU / MPS / CUDA portability

- `model_manager.py` selects `cuda` → `cpu` (Apple MPS is detected but skipped
  because the model uses `torch.float64` ops + `bfloat16` matmuls that crash on
  MPS internally; see in-code notes).
- `models/tools/wan_model.py` and `wan_model_cross_rope.py` were patched to use
  `torch.float32` instead of `torch.float64` for the RoPE / sinusoidal embedding
  ops so the model can run end-to-end on MPS without falling through to CPU.
- `models/tools/wan_model.py`'s self-attention paths now call the `attention(...)`
  wrapper that has a PyTorch SDPA fallback, instead of `flash_attention(...)`
  which asserts `q.device.type == "cuda"`.
- `models/tools/t5.py:T5EncoderModel` no longer evaluates
  `torch.cuda.current_device()` as a default argument at import time, so the
  module imports cleanly on CPU-only PyTorch builds.

### Telemetry harness (kept in the deployed image; harmless)

- `telemetry.py` — append-only JSONL log of session events (`update_text_in`,
  `begin_action`, `frame_added`, `action_done`, `pose_state_change`,
  `transition_inserted`, `pending_motion_dequeued`, `warmup_done`, ...).
- `/api/telemetry` endpoint — lets the browser tag client-side events into the
  same log (`client_chip_click`, `client_fetch_start/done`,
  `client_frame_displayed`).
- `test_text_switch.sh`, `test_intense.sh`, `test_chip_benchmark.sh`,
  `test_multi_switch.sh` — CLI tests that exercise the API directly (no browser
  needed) and produce per-prompt motion statistics. Used during development to
  verify each chip's motion characteristics match the prompt.

## Replication notes

The model itself is unchanged from `ShandaAI/FloodDiffusionTiny`. All edits are
either:

- **Application-level** (`app.py`, `model_manager.py`, `static/`, `templates/`)
  for UX, lifecycle, and rendering.
- **Portability patches** in the Hugging Face cache copy of the model code
  (`ldf_models/tools/{attention,t5,wan_model,wan_model_cross_rope}.py`) — these
  are applied automatically at runtime by `model_manager._patch_attention_sdpa`,
  so a fresh pull of the model on a non-CUDA host works out of the box.

The original repository: [`ShandaAI/FloodDiffusion`](https://github.com/ShandaAI/FloodDiffusion).
The Tiny model card: [`ShandaAI/FloodDiffusionTiny`](https://huggingface.co/ShandaAI/FloodDiffusionTiny).

## Running locally

```bash
# Python 3.10 conda env recommended
pip install -r requirements.txt
python app.py --port 5050
# open http://localhost:5050
```

## License

Apache-2.0 — same as the upstream `FloodDiffusion` codebase. Soldier.glb is the
Three.js sample asset (CC-BY 4.0 via Khronos / mixamo).
