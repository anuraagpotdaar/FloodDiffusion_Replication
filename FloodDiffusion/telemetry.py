"""Append-only JSONL telemetry log. One line per event. Safe to call from any
thread; small writes are serialized by a module-level lock.

Also exposes a DebugLogger used by the --debug flag to write a single
self-contained debug_log_for_anuraag_<timestamp>.log at the repo root with
system/runtime/per-step telemetry and a final summary on shutdown."""
import atexit
import datetime
import json
import os
import threading
import time

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry.jsonl")
_lock = threading.Lock()


def reset():
    with _lock:
        try:
            open(LOG_PATH, "w").close()
        except Exception:
            pass


def log(src, **kwargs):
    event = {"t": time.time(), "src": src}
    event.update(kwargs)
    line = json.dumps(event, default=_default)
    with _lock:
        try:
            with open(LOG_PATH, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
    # Tee per-step events into the debug log if active. Single boolean check
    # in the off path — zero cost when --debug is not set.
    if _DEBUG is not None:
        _DEBUG.record_event(src, event)


def _default(obj):
    try:
        return float(obj)
    except Exception:
        return str(obj)


# ---------------------------------------------------------------------------
# Debug logger
# ---------------------------------------------------------------------------

# Repo root: parent of FloodDiffusion/. Debug artifacts land there per spec.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Events teed into the debug step log. Every gen_step_end is captured per spec.
_TEE_EVENTS = {
    "gen_step_end", "gen_step_start", "server_frames_served",
    "begin_action", "warmup_done", "action_done", "transition_inserted",
    "pose_state_change", "pending_motion_dequeued",
    "server_session_start", "server_update_request",
}

_DEBUG = None  # set by start_debug()


def _timestamp_tag():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


class DebugLogger:
    """Owns the debug_log_for_anuraag_<ts>.log file. All methods are
    best-effort — a logging failure must never crash the app."""

    def __init__(self, repo_root=_REPO_ROOT, profile_enabled=False):
        tag = _timestamp_tag()
        self.log_path = os.path.join(repo_root, f"debug_log_for_anuraag_{tag}.log")
        self.steps_path = os.path.join(
            repo_root, f"debug_log_for_anuraag_{tag}.steps.jsonl"
        )
        self.trace_path = os.path.join(
            repo_root, f"debug_log_for_anuraag_{tag}.trace.json"
        )
        self.profile_enabled = profile_enabled
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._step_durations_ms = []
        self._step_n_frames = []
        self._buf_sizes = []
        self._underrun_count = 0
        self._frames_served = 0
        self._requests = 0
        self._thermal_samples = []  # (t_offset, raw_pmset_output)
        self._thermal_thread = None
        self._stopped = False

    # -- writing helpers ---------------------------------------------------

    def _write(self, text):
        try:
            with self._lock:
                with open(self.log_path, "a") as f:
                    f.write(text)
        except Exception:
            pass

    def _write_section(self, title, body):
        banner = "\n" + "=" * 78 + f"\n=== {title}\n" + "=" * 78 + "\n"
        if isinstance(body, dict):
            body_text = "\n".join(f"  {k}: {v}" for k, v in body.items())
        else:
            body_text = str(body)
        self._write(banner + body_text + "\n")

    # -- public API --------------------------------------------------------

    def write_static_snapshot(self):
        """Sections A, B, env vars, pip freeze. Call once at startup."""
        from debug_utils import (
            system_snapshot, torch_runtime_snapshot, env_snapshot, pip_freeze,
            is_rosetta,
        )
        self._write(
            f"debug_log_for_anuraag — started {time.ctime(self.start_time)}\n"
            f"profile_enabled: {self.profile_enabled}\n"
        )
        self._write_section("A. SYSTEM", system_snapshot())
        self._write_section("B. PYTORCH RUNTIME", torch_runtime_snapshot())
        self._write_section("ENV VARS (relevant subset)", env_snapshot())
        if is_rosetta():
            self._write_section(
                "!! ROSETTA DETECTED",
                "Python is running under x86_64 Rosetta translation. "
                "MPS will not work; native arm64 Python is required.",
            )
        self._write_section("PIP FREEZE", pip_freeze())

    def write_model_snapshot(self, model_manager):
        """Section C. Call once after model load."""
        from debug_utils import model_snapshot
        self._write_section("C. MODEL", model_snapshot(model_manager))

    def start_thermal_sampler(self, interval_s=5.0):
        """Section D's thermal sub-stream. macOS only; no-op elsewhere."""
        from debug_utils import is_macos, thermal_pressure
        if not is_macos():
            return

        def _loop():
            while not self._stopped:
                t_off = time.time() - self.start_time
                self._thermal_samples.append((t_off, thermal_pressure()))
                if len(self._thermal_samples) > 2000:
                    self._thermal_samples = self._thermal_samples[-1000:]
                time.sleep(interval_s)

        self._thermal_thread = threading.Thread(target=_loop, daemon=True)
        self._thermal_thread.start()

    def record_event(self, src, event):
        """Tee from telemetry.log() into .steps.jsonl + update counters."""
        if src not in _TEE_EVENTS:
            return
        try:
            with self._lock:
                with open(self.steps_path, "a") as f:
                    f.write(json.dumps(event, default=_default) + "\n")
        except Exception:
            pass

        if src == "gen_step_end":
            dur = event.get("duration_ms", 0.0)
            n = event.get("n_frames", 0)
            buf = event.get("buf_size", 0)
            self._step_durations_ms.append(dur)
            self._step_n_frames.append(n)
            self._buf_sizes.append(buf)
            if buf == 0:
                self._underrun_count += 1
        elif src == "server_frames_served":
            self._requests += 1
            self._frames_served += event.get("n", 0)

    def write_profiler_summary(self, key_averages_table, trace_written):
        """Section F. Called from model_manager after the profiler runs."""
        body = (
            f"profile_window_complete: {bool(trace_written)}\n"
            f"trace_path: {self.trace_path if trace_written else '<not written>'}\n\n"
            f"{key_averages_table}\n"
        )
        self._write_section("F. TORCH PROFILER (top ops by self time)", body)

    def finalize(self):
        """Section G. Print final summary on shutdown."""
        if self._stopped:
            return
        self._stopped = True
        try:
            elapsed = time.time() - self.start_time
            durs = sorted(self._step_durations_ms)
            n_steps = len(durs)
            summary = {
                "elapsed_s": round(elapsed, 2),
                "n_steps": n_steps,
                "n_frames_served": self._frames_served,
                "n_requests": self._requests,
                "buffer_underrun_count": self._underrun_count,
            }
            if n_steps > 0:
                summary["step_ms_p50"] = round(_pct(durs, 0.50), 2)
                summary["step_ms_p95"] = round(_pct(durs, 0.95), 2)
                summary["step_ms_p99"] = round(_pct(durs, 0.99), 2)
                summary["step_ms_max"] = round(durs[-1], 2)
                total_frames = sum(self._step_n_frames)
                gen_time_s = sum(self._step_durations_ms) / 1000.0
                if gen_time_s > 0:
                    summary["avg_fps"] = round(total_frames / gen_time_s, 2)
                if self._buf_sizes:
                    summary["buf_size_avg"] = round(
                        sum(self._buf_sizes) / len(self._buf_sizes), 2
                    )
                    summary["buf_size_min"] = min(self._buf_sizes)
            self._write_section("G. FINAL SUMMARY", summary)

            if self._thermal_samples:
                # Only emit lines where pmset output changed — baseline
                # CPU_Speed_Limit = 100 is uninteresting; deviation = throttle.
                lines = ["t_offset_s | pmset -g therm output"]
                last = None
                for t_off, raw in self._thermal_samples:
                    if raw != last:
                        lines.append(f"  {t_off:7.1f} | {raw}")
                        last = raw
                lines.append(f"  (total samples: {len(self._thermal_samples)})")
                self._write_section("THERMAL TIMELINE", "\n".join(lines))
        except Exception as e:
            self._write_section("FINAL SUMMARY ERROR", str(e))


def _pct(sorted_list, q):
    if not sorted_list:
        return 0.0
    idx = max(0, min(len(sorted_list) - 1, int(q * len(sorted_list))))
    return sorted_list[idx]


def start_debug(profile_enabled=False):
    """Enable debug logging. Idempotent. Returns the active DebugLogger."""
    global _DEBUG
    if _DEBUG is not None:
        return _DEBUG
    _DEBUG = DebugLogger(profile_enabled=profile_enabled)
    _DEBUG.write_static_snapshot()
    _DEBUG.start_thermal_sampler()
    atexit.register(_DEBUG.finalize)
    return _DEBUG


def get_debug():
    """Return the active DebugLogger or None."""
    return _DEBUG
