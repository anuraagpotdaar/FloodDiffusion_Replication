"""
Model Manager for real-time motion generation (HF Space version)
Loads model from Hugging Face Hub instead of local checkpoints.
"""

import threading
import time
from collections import deque

import numpy as np
import torch

from motion_process import StreamJointRecovery263
import telemetry

# Set by app.py when --debug --profile is passed. When True, the generation
# loop wraps PROFILE_STEPS steps after warmup with torch.profiler and dumps
# a Chrome trace + an op-summary table into the debug log.
PROFILE_REQUESTED = False
PROFILE_WARMUP_STEPS = 10
PROFILE_STEPS = 100


class FrameBuffer:
    """
    Thread-safe frame buffer that maintains a queue of generated frames
    """

    def __init__(self, target_buffer_size=4):
        self.buffer = deque(maxlen=100)  # Max 100 frames in buffer
        self.target_size = target_buffer_size
        self.lock = threading.Lock()

    def add_frame(self, joints):
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(joints)

    def get_frame(self):
        """Get the next frame from buffer"""
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            return None

    def size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)

    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()

    def needs_generation(self):
        """Check if buffer needs more frames"""
        return self.size() < self.target_size


class ModelManager:
    """
    Manages model loading from HF Hub and real-time frame generation
    """

    def __init__(self, model_name):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Load models from HF Hub
        self.vae, self.model = self._load_models(model_name)

        # Build config dicts from model's individual attributes (HF model API)
        self._base_schedule_config = {
            "chunk_size": self.model.chunk_size,
            "steps": self.model.noise_steps,
        }
        self._base_cfg_config = {
            "cfg_scale": self.model.cfg_scale,
        }

        # Frame buffer (for active session). Smaller target = fresher response to
        # text updates at the cost of more sensitivity to generation stalls.
        self.frame_buffer = FrameBuffer(target_buffer_size=4)

        # Broadcast buffer (for spectators) - append-only with frame IDs
        self.broadcast_frames = deque(maxlen=200)
        self.broadcast_id = 0
        self.broadcast_lock = threading.Lock()

        # Stream joint recovery with smoothing
        self.smoothing_alpha = 0.5  # Default: medium smoothing
        self.stream_recovery = StreamJointRecovery263(
            joints_num=22, smoothing_alpha=self.smoothing_alpha
        )

        # Generation state
        self.current_text = ""
        self.is_generating = False
        self.generation_thread = None
        self.should_stop = False

        # Model generation state
        self.first_chunk = True  # For VAE stream_decode
        self._model_first_chunk = True  # For model stream_generate_step
        self.history_length = 30

        # Per-action lifecycle. Each chip click / text update starts a new action;
        # generation runs until either (a) max_frames is reached or (b) motion has
        # been "settled" (per-frame joint displacement < threshold) for a sustained
        # window. min_frames guarantees the action gets to play out before settling
        # detection kicks in. Continuous motions (walk, dance) hit the max cap;
        # one-shot motions (sit, jump, wave) usually finish on the settle path.
        self.action_min_frames = 80       # ~4 s @ 20 FPS — guaranteed playback
        self.action_max_frames = 240      # ~12 s — hard cap for continuous motions
        self.stillness_threshold = 0.30   # sum of 22 per-joint deltas (m)
        self.stillness_window = 16        # ~0.8 s window of low motion → "done"
        self._frames_in_action = 0
        self._motion_history = []
        # Warmup: after init_generated re-randomises the latents, the model needs
        # ~one chunk to converge to clean motion under the new prompt. Those
        # warmup frames look glitchy (e.g. character "falls" when switching from
        # run → stand) so we silently drop them — the buffer only receives clean
        # frames produced after the model has settled into the new prompt.
        self.warmup_frames = 5
        self._warmup_remaining = 0
        # Motion catalog. type=continuous loops until the user interrupts;
        # type=oneshot plays once and ends in the listed pose. needs_starting
        # gates the motion behind a transition prompt if the current pose
        # doesn't match (e.g. clicking "walk" while sitting → first auto-plays
        # "stand up", then "walk").
        self.motions = {
            "walk forward":     {"type": "continuous", "ends_in": "standing",  "needs_starting": "standing"},
            "run forward":      {"type": "continuous", "ends_in": "standing",  "needs_starting": "standing"},
            "dance":            {"type": "continuous", "ends_in": "standing",  "needs_starting": "standing"},
            "walk in a circle": {"type": "continuous", "ends_in": "standing",  "needs_starting": "standing"},
            "turn around":      {"type": "continuous", "ends_in": "standing",  "needs_starting": "standing"},
            "jump":             {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "standing"},
            "sit down":         {"type": "oneshot",    "ends_in": "sitting",   "needs_starting": "standing"},
            "stand up":         {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "any"},
            "wave hand":        {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "standing"},
            "kick":             {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "standing"},
            "punch":            {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "standing"},
            "crouch":           {"type": "oneshot",    "ends_in": "crouching", "needs_starting": "standing"},
            "clap hands":       {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "standing"},
            "bow":              {"type": "oneshot",    "ends_in": "standing",  "needs_starting": "standing"},
        }
        self._pose_state = "standing"     # what pose the character is in right now
        self._pending_motion = None       # (text, meta) queued behind a transition
        self._motion_type = "oneshot"     # type of the current action

        print("ModelManager initialized successfully")

    def _patch_attention_sdpa(self, model_name):
        """Patch flash_attention() to include SDPA fallback for GPUs without flash-attn (e.g., T4)."""
        import glob
        import os

        hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        patterns = [
            os.path.join(
                hf_cache, "hub", "models--" + model_name.replace("/", "--"),
                "snapshots", "*", "ldf_models", "tools", "attention.py",
            ),
            os.path.join(
                hf_cache, "modules", "transformers_modules", model_name,
                "*", "ldf_models", "tools", "attention.py",
            ),
        ]

        # Use the assert + next line as target to ensure idempotent patching
        target = (
            '    assert q.device.type == "cuda" and q.size(-1) <= 256\n'
            "\n"
            "    # params\n"
        )
        replacement = (
            '    assert q.size(-1) <= 256  # cuda check relaxed for MPS/CPU\n'
            "\n"
            "    # SDPA fallback when flash-attn is not available (e.g., T4 GPU, MPS, CPU)\n"
            "    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:\n"
            "        out_dtype = q.dtype\n"
            "        b, lq, nq, c = q.shape\n"
            "        lk = k.size(1)\n"
            "        q = q.transpose(1, 2).to(dtype)\n"
            "        k = k.transpose(1, 2).to(dtype)\n"
            "        v = v.transpose(1, 2).to(dtype)\n"
            "        attn_mask = None\n"
            "        is_causal_flag = causal\n"
            "        if k_lens is not None:\n"
            "            k_lens = k_lens.to(q.device)\n"
            "            valid = torch.arange(lk, device=q.device).unsqueeze(0) < k_lens.unsqueeze(1)\n"
            "            attn_mask = torch.where(valid[:, None, None, :], 0.0, float('-inf')).to(dtype=dtype)\n"
            "            is_causal_flag = False\n"
            "            if causal:\n"
            "                cm = torch.triu(torch.ones(lq, lk, device=q.device, dtype=torch.bool), diagonal=1)\n"
            "                attn_mask = attn_mask.masked_fill(cm[None, None, :, :], float('-inf'))\n"
            "        out = torch.nn.functional.scaled_dot_product_attention(\n"
            "            q, k, v, attn_mask=attn_mask, is_causal=is_causal_flag, dropout_p=dropout_p\n"
            "        )\n"
            "        return out.transpose(1, 2).contiguous().to(out_dtype)\n"
            "\n"
            "    # params\n"
        )

        for pattern in patterns:
            for filepath in glob.glob(pattern):
                with open(filepath, "r") as f:
                    content = f.read()
                if "SDPA fallback" in content:
                    print(f"Already patched: {filepath}")
                    continue
                if target in content:
                    content = content.replace(target, replacement, 1)
                    with open(filepath, "w") as f:
                        f.write(content)
                    print(f"Patched with SDPA fallback: {filepath}")

    def _load_models(self, model_name):
        """Load VAE and diffusion models from HF Hub"""
        torch.set_float32_matmul_precision("high")

        # Pre-download model files to hub cache
        print(f"Downloading model from HF Hub: {model_name}")
        from huggingface_hub import snapshot_download
        snapshot_download(model_name)

        # Patch flash_attention with SDPA fallback for T4 (no flash-attn)
        self._patch_attention_sdpa(model_name)

        print("Loading model...")
        from transformers import AutoModel

        hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hf_model.to(self.device)

        # Trigger lazy loading / warmup
        print("Warming up model...")
        _ = hf_model("test", length=1)

        # Access underlying streaming components
        model = hf_model.ldf_model
        vae = hf_model.vae

        model.eval()
        vae.eval()

        print("Models loaded successfully")
        return vae, model

    def start_generation(self, text, history_length=None):
        """Start or update generation with new text"""
        self.current_text = text

        if history_length is not None:
            self.history_length = history_length

        if not self.is_generating:
            # Reset state before starting (only once at the beginning)
            self.frame_buffer.clear()
            self.stream_recovery.reset()
            self.vae.clear_cache()
            self.first_chunk = True
            self._model_first_chunk = True
            # Restore model params from base config
            self.model.chunk_size = self._base_schedule_config["chunk_size"]
            self.model.noise_steps = self._base_schedule_config["steps"]
            self.model.cfg_scale = self._base_cfg_config["cfg_scale"]
            self._pose_state = "standing"   # character starts standing
            self._pending_motion = None
            # _begin_action handles init_generated + warmup + motion-type +
            # _target_end_pose so we don't duplicate that logic here.
            self._begin_action(text, self._lookup_motion(text))
            print(
                f"Model initialized with history length: {self.history_length}, "
                f"action limits: min={self.action_min_frames}, max={self.action_max_frames}"
            )

            # Start generation thread
            self.should_stop = False
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            self.is_generating = True

    def _lookup_motion(self, text):
        """Get motion metadata for a prompt. Match is case-insensitive and
        ignores trailing punctuation/whitespace, so "Walk in a circle." resolves
        to the same entry as "walk in a circle". Unknown prompts default to
        continuous (so free-text prompts loop instead of auto-ending after the
        max cap)."""
        if text:
            norm = text.strip().rstrip('.,!?;:').lower()
            for key, meta in self.motions.items():
                if key.lower() == norm:
                    return meta
        return {"type": "continuous", "ends_in": "standing", "needs_starting": "any"}

    def update_text(self, text):
        """Pose-aware update. If the requested motion needs the character in a
        specific starting pose (e.g. walking needs standing) and the character
        isn't in that pose (e.g. currently sitting), queue the requested motion
        behind a transition prompt ("stand up") so the character first stands,
        then plays the chosen motion."""
        telemetry.log("update_text_in", new=text, old=self.current_text,
                      pose=self._pose_state)
        if text == self.current_text:
            telemetry.log("update_text_skip", reason="unchanged", text=text)
            return
        meta = self._lookup_motion(text)
        # If the character isn't in the right starting pose, insert a transition.
        # Currently any non-standing pose recovers via "stand up".
        needs = meta.get("needs_starting", "any")
        if needs != "any" and self._pose_state != needs:
            transition = "stand up" if needs == "standing" else None
            if transition and transition in self.motions:
                self._pending_motion = (text, meta)
                telemetry.log("transition_inserted", from_pose=self._pose_state,
                              to_pose=needs, transition=transition,
                              queued=text)
                self._begin_action(transition, self._lookup_motion(transition))
                return
        # No transition needed — play directly.
        self._pending_motion = None
        self._begin_action(text, meta)

    def _begin_action(self, text, meta):
        """Reset latents + cond list + warmup window for a new action."""
        old = self.current_text
        self.current_text = text
        self._motion_type = meta["type"]
        self._target_end_pose = meta.get("ends_in", "standing")
        try:
            self.model.init_generated(
                self.history_length,
                batch_size=1,
                num_denoise_steps=self.model.num_denoise_steps,
            )
            self.vae.clear_cache()
            self.first_chunk = True
            self._model_first_chunk = True
            self._frames_in_action = 0
            self._motion_history = []
            self._warmup_remaining = self.warmup_frames
            telemetry.log("begin_action", old=old, new=text,
                          motion_type=self._motion_type,
                          ends_in=self._target_end_pose,
                          pose_state=self._pose_state)
        except Exception as e:
            telemetry.log("begin_action_error", error=str(e))

    def pause_generation(self):
        """Pause generation (keeps all state)"""
        self.should_stop = True
        if self.generation_thread:
            self.generation_thread.join(timeout=2.0)
        self.is_generating = False
        print("Generation paused (state preserved)")

    def resume_generation(self):
        """Resume generation from paused state"""
        if self.is_generating:
            print("Already generating, ignoring resume")
            return

        # Restart generation thread with existing state
        self.should_stop = False
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        self.is_generating = True
        print("Generation resumed")

    def reset(self, history_length=None, smoothing_alpha=None):
        """Reset generation state completely

        Args:
            history_length: History window length for the model
            smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
                - 1.0 = no smoothing (default)
                - 0.0 = infinite smoothing
                - Recommended: 0.3-0.7 for visible smoothing
        """
        # Stop if running
        if self.is_generating:
            self.pause_generation()

        # Clear everything
        self.frame_buffer.clear()
        self.vae.clear_cache()
        self.first_chunk = True

        if history_length is not None:
            self.history_length = history_length

        # Update smoothing alpha if provided and recreate stream recovery
        if smoothing_alpha is not None:
            self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
            print(f"Smoothing alpha updated to: {self.smoothing_alpha}")

        # Recreate stream recovery with new smoothing alpha
        self.stream_recovery = StreamJointRecovery263(
            joints_num=22, smoothing_alpha=self.smoothing_alpha
        )

        # Restore model params from base config
        self.model.chunk_size = self._base_schedule_config["chunk_size"]
        self.model.noise_steps = self._base_schedule_config["steps"]
        self.model.cfg_scale = self._base_cfg_config["cfg_scale"]
        self._model_first_chunk = True

        # Initialize model
        self.model.init_generated(self.history_length, batch_size=1)
        print(
            f"Model reset - history: {self.history_length}, smoothing: {self.smoothing_alpha}"
        )

    def _is_action_done(self):
        """Continuous motions never auto-end (they run until the user clicks
        another chip). One-shot motions end when settled or at the max cap.
        """
        if self._motion_type == "continuous":
            return False, None
        max_cap = self.action_max_frames
        min_cap = self.action_min_frames
        if self._frames_in_action >= max_cap:
            return True, "max_frames"
        if (self._frames_in_action >= min_cap
                and len(self._motion_history) > self.stillness_window):
            window = self._motion_history[-(self.stillness_window + 1):]
            max_delta = 0.0
            for i in range(1, len(window)):
                d = float(np.linalg.norm(window[i] - window[i - 1], axis=1).sum())
                if d > max_delta:
                    max_delta = d
                if max_delta >= self.stillness_threshold:
                    return False, None  # still moving — keep generating
            return True, f"settled (max_delta={max_delta:.3f})"
        return False, None

    def _on_action_settled(self):
        """A one-shot action just finished. Update pose state to match what the
        action ends in, and if there's a queued motion (we just finished a
        transition like "stand up"), kick it off now."""
        ended_in = getattr(self, "_target_end_pose", "standing")
        self._pose_state = ended_in
        telemetry.log("pose_state_change", pose=ended_in)
        if self._pending_motion is not None:
            text, meta = self._pending_motion
            self._pending_motion = None
            telemetry.log("pending_motion_dequeued", text=text)
            self._begin_action(text, meta)

    def _generation_loop(self):
        """Main generation loop that runs in background thread"""
        print("Generation loop started")

        step_count = 0
        total_gen_time = 0

        # Profiler state. Activates once after warmup completes; runs for
        # PROFILE_STEPS, then dumps the trace and disables itself.
        prof = None
        prof_steps_remaining = 0
        prof_done = False

        def _maybe_start_profiler():
            nonlocal prof, prof_steps_remaining, prof_done
            if not PROFILE_REQUESTED or prof is not None or prof_done:
                return
            if step_count < PROFILE_WARMUP_STEPS:
                return
            try:
                from torch.profiler import profile, ProfilerActivity
                acts = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    acts.append(ProfilerActivity.CUDA)
                prof = profile(activities=acts, record_shapes=False)
                prof.__enter__()
                prof_steps_remaining = PROFILE_STEPS
                print(f"[profile] started after warmup; capturing {PROFILE_STEPS} steps")
            except Exception as e:
                print(f"[profile] failed to start: {e}")
                prof_done = True

        def _maybe_finish_profiler():
            nonlocal prof, prof_steps_remaining, prof_done
            if prof is None or prof_steps_remaining > 0:
                return
            try:
                prof.__exit__(None, None, None)
                table = prof.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=20
                )
                trace_written = False
                dbg = telemetry.get_debug()
                if dbg is not None:
                    try:
                        prof.export_chrome_trace(dbg.trace_path)
                        trace_written = True
                    except Exception as e:
                        print(f"[profile] export_chrome_trace failed: {e}")
                    dbg.write_profiler_summary(table, trace_written)
                print("[profile] window complete; trace dumped")
            except Exception as e:
                print(f"[profile] finalize error: {e}")
            finally:
                prof = None
                prof_done = True

        last_action_done_logged = False
        with torch.no_grad():
            while not self.should_stop:
                # One-shot motions settle (or hit max) → update pose state, run
                # any queued motion (transition completion), else idle.
                # Continuous motions never auto-end; they run until update_text.
                action_done, done_reason = self._is_action_done()
                if action_done and not last_action_done_logged:
                    telemetry.log("action_done", reason=done_reason,
                                  frames_in_action=self._frames_in_action,
                                  text=self.current_text,
                                  motion_type=self._motion_type)
                    self._on_action_settled()
                    last_action_done_logged = True
                elif not action_done:
                    last_action_done_logged = False
                if self.frame_buffer.needs_generation() and not action_done:
                    try:
                        step_start = time.time()

                        # Generate one token (produces frames from VAE)
                        snapshot_text = self.current_text
                        was_first_chunk = self._model_first_chunk
                        x = {"text": [snapshot_text]}
                        telemetry.log("gen_step_start", step=step_count,
                                      text=snapshot_text, first_chunk=was_first_chunk,
                                      tcl_len=len(self.model.text_condition_list[0]))

                        # Generate from model (1 token)
                        output = self.model.stream_generate_step(
                            x, first_chunk=was_first_chunk
                        )
                        self._model_first_chunk = False
                        generated = output["generated"]

                        # Skip if no frames committed yet
                        if generated[0].shape[0] == 0:
                            continue

                        # Decode with VAE (1 token -> 4 frames)
                        decoded = self.vae.stream_decode(
                            generated[0][None, :], first_chunk=self.first_chunk
                        )[0]

                        self.first_chunk = False

                        # Convert each frame to joints
                        for i in range(decoded.shape[0]):
                            frame_data = decoded[i].cpu().numpy()
                            # Discard the chunk's worth of warmup frames after
                            # any latent reset (init_generated). Skip stream
                            # recovery too — feeding it random-noise velocities
                            # would corrupt its accumulated trajectory.
                            if self._warmup_remaining > 0:
                                self._warmup_remaining -= 1
                                if self._warmup_remaining == 0:
                                    telemetry.log("warmup_done", text=snapshot_text)
                                continue
                            joints = self.stream_recovery.process_frame(frame_data)
                            self.frame_buffer.add_frame(joints)
                            # Also add to broadcast buffer for spectators
                            with self.broadcast_lock:
                                self.broadcast_id += 1
                                self.broadcast_frames.append(
                                    (self.broadcast_id, joints)
                                )
                            telemetry.log("frame_added", step=step_count,
                                          text=snapshot_text, frame_idx=i,
                                          root=[float(joints[0][0]),
                                                float(joints[0][1]),
                                                float(joints[0][2])])
                            # Bookkeeping for action-completion detection
                            self._frames_in_action += 1
                            self._motion_history.append(joints)
                            if len(self._motion_history) > self.stillness_window + 2:
                                self._motion_history.pop(0)

                        step_time = time.time() - step_start
                        total_gen_time += step_time
                        step_count += 1
                        if prof is not None and prof_steps_remaining > 0:
                            prof_steps_remaining -= 1
                            _maybe_finish_profiler()
                        else:
                            _maybe_start_profiler()
                        n_produced = int(decoded.shape[0])
                        telemetry.log("gen_step_end", step=step_count,
                                      text=snapshot_text,
                                      n_frames=n_produced,
                                      duration_ms=step_time * 1000.0,
                                      buf_size=self.frame_buffer.size(),
                                      frames_in_action=self._frames_in_action)

                        # Print performance stats every 10 steps
                        if step_count % 10 == 0:
                            avg_time = total_gen_time / step_count
                            fps = decoded.shape[0] / avg_time
                            print(
                                f"[Generation] Step {step_count}: {step_time * 1000:.1f}ms, "
                                f"Avg: {avg_time * 1000:.1f}ms, "
                                f"FPS: {fps:.1f}, "
                                f"Buffer: {self.frame_buffer.size()}"
                            )

                    except Exception as e:
                        print(f"Error in generation: {e}")
                        import traceback

                        traceback.print_exc()
                        time.sleep(0.1)
                else:
                    # Buffer is full, wait a bit
                    time.sleep(0.01)

        # If profiler was mid-capture when the loop exited, force-finalize so
        # the trace file isn't leaked.
        if prof is not None:
            prof_steps_remaining = 0
            _maybe_finish_profiler()

        print("Generation loop stopped")

    def get_next_frame(self):
        """Get the next frame from buffer"""
        return self.frame_buffer.get_frame()

    def get_broadcast_frames(self, after_id, count=8):
        """Get frames from broadcast buffer after the given ID (for spectators)."""
        with self.broadcast_lock:
            frames = [
                (fid, joints)
                for fid, joints in self.broadcast_frames
                if fid > after_id
            ]
        return frames[:count]

    def get_buffer_status(self):
        """Get buffer status"""
        return {
            "buffer_size": self.frame_buffer.size(),
            "target_size": self.frame_buffer.target_size,
            "is_generating": self.is_generating,
            "current_text": self.current_text,
            "smoothing_alpha": self.smoothing_alpha,
            "history_length": self.history_length,
            "schedule_config": {
                "chunk_size": self.model.chunk_size,
                "steps": self.model.noise_steps,
            },
            "cfg_config": {
                "cfg_scale": self.model.cfg_scale,
            },
        }


# Global model manager instance
_model_manager = None


def get_model_manager(model_name=None):
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(model_name)
    return _model_manager
