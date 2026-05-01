"""Best-effort system snapshot helpers for the --debug flag.

All shell-outs are wrapped so a missing command never crashes the app.
Stdlib-only; safe to import on any platform.
"""
import os
import platform
import shutil
import subprocess
import sys


def _run(cmd, timeout=2.0):
    """Run a shell command, return stripped stdout or 'n/a' on any failure."""
    try:
        args = cmd.split() if isinstance(cmd, str) else list(cmd)
        if not shutil.which(args[0]):
            return "n/a (command not found)"
        out = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        text = (out.stdout or out.stderr or "").strip()
        return text if text else "n/a (empty output)"
    except Exception as e:
        return f"n/a ({type(e).__name__}: {e})"


def is_macos():
    return sys.platform == "darwin"


def is_rosetta():
    """True if Python is running under Rosetta translation on Apple Silicon."""
    if not is_macos():
        return False
    try:
        out = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=1.0, check=False,
        )
        return out.stdout.strip() == "1"
    except Exception:
        return False


def system_snapshot():
    """Return a dict of static system info. Keys are stable for diff'ing."""
    snap = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "n/a",
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "rosetta": is_rosetta(),
    }
    if is_macos():
        snap.update({
            "macos_version": _run(["sw_vers", "-productVersion"]),
            "macos_build": _run(["sw_vers", "-buildVersion"]),
            "cpu_brand": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "cpu_cores_total": _run(["sysctl", "-n", "hw.ncpu"]),
            "perf_cores": _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
            "efficiency_cores": _run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]),
            "memsize_bytes": _run(["sysctl", "-n", "hw.memsize"]),
            "gpu_info": _run(["system_profiler", "SPDisplaysDataType"], timeout=8.0),
            "power_source": _run(["pmset", "-g", "ps"]),
            "power_settings": _run(["pmset", "-g"]),
            "thermal_pressure": _run(["pmset", "-g", "therm"]),
            "vm_stat": _run(["vm_stat"]),
        })
    return snap


def thermal_pressure():
    """Quick thermal sample for periodic logging. macOS only; '' elsewhere."""
    if not is_macos():
        return ""
    return _run(["pmset", "-g", "therm"], timeout=1.0)


def torch_runtime_snapshot():
    """PyTorch / device facts. Imports torch lazily."""
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_git"] = getattr(torch.version, "git_version", "n/a")
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()
        info["matmul_precision"] = torch.get_float32_matmul_precision()
        info["default_dtype"] = str(torch.get_default_dtype())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = str(torch.cuda.get_device_capability(0))
        if torch.backends.mps.is_available():
            try:
                info["mps_current_alloc_bytes"] = torch.mps.current_allocated_memory()
                info["mps_driver_alloc_bytes"] = torch.mps.driver_allocated_memory()
            except Exception as e:
                info["mps_alloc_error"] = str(e)
    except Exception as e:
        info["torch_import_error"] = str(e)
    return info


def env_snapshot():
    """Subset of env vars that affect MPS/threading performance."""
    keys = [
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYTHONUNBUFFERED",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TOKENIZERS_PARALLELISM",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


def pip_freeze():
    """pip freeze output. Verbose but invaluable for cross-machine diff."""
    return _run([sys.executable, "-m", "pip", "freeze"], timeout=15.0)


def model_snapshot(model_manager):
    """Counts and footprint for the loaded model."""
    info = {}
    try:
        import torch
        info["device"] = model_manager.device
        info["target_buffer_size"] = model_manager.frame_buffer.target_size
        info["history_length"] = model_manager.history_length
        info["smoothing_alpha"] = float(model_manager.smoothing_alpha)
        info["chunk_size"] = model_manager.model.chunk_size
        info["noise_steps"] = model_manager.model.noise_steps
        info["cfg_scale"] = model_manager.model.cfg_scale

        n_params = sum(p.numel() for p in model_manager.model.parameters())
        info["model_param_count"] = n_params
        dtypes = {}
        for p in model_manager.model.parameters():
            k = str(p.dtype)
            dtypes[k] = dtypes.get(k, 0) + p.numel()
        info["model_dtype_histogram"] = dtypes

        n_vae = sum(p.numel() for p in model_manager.vae.parameters())
        info["vae_param_count"] = n_vae

        if model_manager.device == "mps":
            try:
                info["mps_current_alloc_bytes"] = torch.mps.current_allocated_memory()
                info["mps_driver_alloc_bytes"] = torch.mps.driver_allocated_memory()
            except Exception as e:
                info["mps_alloc_error"] = str(e)
    except Exception as e:
        info["snapshot_error"] = str(e)
    return info
