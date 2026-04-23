"""
Flask server for real-time 3D motion generation demo (HF Space version)
"""

import argparse
import threading
import time

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from model_manager import get_model_manager
import telemetry


def _coerce_value(value, reference):
    """Coerce a value to match the type of a reference value"""
    if isinstance(reference, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("true", "1")
    elif isinstance(reference, int):
        return int(value)
    elif isinstance(reference, float):
        return float(value)
    return str(value)


app = Flask(__name__)
CORS(app)

# Global model manager (loaded eagerly on startup)
model_manager = None
model_name_global = None  # Will be set once at startup

# Session tracking - only one active session can generate at a time
active_session_id = None  # The session ID currently generating
session_lock = threading.Lock()

# Frame consumption monitoring - detect if client disconnected by tracking frame consumption
last_frame_consumed_time = None
consumption_timeout = (
    60.0  # 5 s was too aggressive — slow generation on CPU + a hidden browser tab
    # could reset the session mid-interaction, causing /api/update_text to fail
    # silently with 403 (and the model to seem to "ignore" prompts).
)
consumption_monitor_thread = None
consumption_monitor_lock = threading.Lock()


def init_model():
    """Initialize model manager"""
    global model_manager
    if model_manager is None:
        if model_name_global is None:
            raise RuntimeError(
                "model_name_global not set. Server not properly initialized."
            )
        print(f"Initializing model manager with model: {model_name_global}")
        model_manager = get_model_manager(model_name=model_name_global)
        print("Model manager ready!")
    return model_manager


def consumption_monitor():
    """Monitor frame consumption and auto-reset if client stops consuming"""
    global last_frame_consumed_time, active_session_id, model_manager

    while True:
        time.sleep(2.0)  # Check every 2 seconds

        # Read state with proper locking - no nested locks!
        should_reset = False
        current_session = None
        time_since_last_consumption = 0

        # First, check consumption time
        with consumption_monitor_lock:
            if last_frame_consumed_time is not None:
                time_since_last_consumption = time.time() - last_frame_consumed_time
                if time_since_last_consumption > consumption_timeout:
                    # Need to check if still generating before reset
                    if model_manager and model_manager.is_generating:
                        should_reset = True

        # Then, get current session (separate lock)
        if should_reset:
            with session_lock:
                current_session = active_session_id

        # Perform reset outside of locks to avoid deadlock
        if should_reset and current_session is not None:
            print(
                f"No frame consumed for {time_since_last_consumption:.1f}s - client disconnected, auto-resetting..."
            )

            if model_manager:
                model_manager.reset()
                print(
                    "Generation reset due to client disconnect (no frame consumption)"
                )

            # Clear state with proper locking - no nested locks!
            with session_lock:
                if active_session_id == current_session:
                    active_session_id = None

            with consumption_monitor_lock:
                last_frame_consumed_time = None


def start_consumption_monitor():
    """Start the consumption monitoring thread if not already running"""
    global consumption_monitor_thread

    if consumption_monitor_thread is None or not consumption_monitor_thread.is_alive():
        consumption_monitor_thread = threading.Thread(
            target=consumption_monitor, daemon=True
        )
        consumption_monitor_thread.start()
        print("Consumption monitor started")


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current config"""
    try:
        if model_manager:
            status = model_manager.get_buffer_status()
            return jsonify(
                {
                    "schedule_config": status["schedule_config"],
                    "cfg_config": status["cfg_config"],
                    "history_length": status["history_length"],
                    "smoothing_alpha": float(status["smoothing_alpha"]),
                }
            )
        else:
            # Model not loaded yet - return defaults
            return jsonify(
                {
                    "schedule_config": {},
                    "cfg_config": {},
                    "history_length": 30,
                    "smoothing_alpha": 0.5,
                }
            )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update model config in memory"""
    try:
        global active_session_id, last_frame_consumed_time

        if not model_manager or not model_manager.model:
            return jsonify({"status": "error", "message": "Model not loaded yet"}), 400

        data = request.json
        new_schedule_config = data.get("schedule_config")
        new_cfg_config = data.get("cfg_config")
        history_length = data.get("history_length")
        smoothing_alpha = data.get("smoothing_alpha")

        valid_schedule_keys = set(model_manager._base_schedule_config.keys())
        valid_cfg_keys = set(model_manager._base_cfg_config.keys())

        # Validate and update schedule_config
        if new_schedule_config:
            for key in new_schedule_config:
                if key not in valid_schedule_keys:
                    return jsonify(
                        {
                            "status": "error",
                            "message": f"Unknown schedule_config key: {key}",
                        }
                    ), 400
            for key, value in new_schedule_config.items():
                model_manager._base_schedule_config[key] = _coerce_value(
                    value, model_manager._base_schedule_config[key]
                )

        # Validate and update cfg_config
        if new_cfg_config:
            for key in new_cfg_config:
                if key not in valid_cfg_keys:
                    return jsonify(
                        {"status": "error", "message": f"Unknown cfg_config key: {key}"}
                    ), 400
            for key, value in new_cfg_config.items():
                model_manager._base_cfg_config[key] = _coerce_value(
                    value, model_manager._base_cfg_config[key]
                )

        # Reset with new parameters
        model_manager.reset(
            history_length=history_length,
            smoothing_alpha=smoothing_alpha,
        )

        # Clear active session
        with session_lock:
            active_session_id = None
        with consumption_monitor_lock:
            last_frame_consumed_time = None

        return jsonify({"status": "success"})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/start", methods=["POST"])
def start_generation():
    """Start generation with given text"""
    try:
        global active_session_id, last_frame_consumed_time

        data = request.json
        session_id = data.get("session_id")
        text = data.get("text", "walk in a circle.")
        history_length = data.get("history_length")
        smoothing_alpha = data.get(
            "smoothing_alpha", None
        )  # Optional smoothing parameter
        force = data.get("force", False)  # Allow force takeover

        if not session_id:
            return jsonify(
                {"status": "error", "message": "session_id is required"}
            ), 400

        print(
            f"[Session {session_id}] Starting generation with text: {text}, history_length: {history_length}, force: {force}"
        )

        # Initialize model if needed
        mm = init_model()

        # Check if another session is already generating
        need_force_takeover = False

        with session_lock:
            if active_session_id and active_session_id != session_id:
                if not force:
                    # Another session is active, return conflict
                    return jsonify(
                        {
                            "status": "error",
                            "message": "Another session is already generating.",
                            "conflict": True,
                            "active_session_id": active_session_id,
                        }
                    ), 409
                else:
                    # Force takeover
                    print(
                        f"[Session {session_id}] Force takeover from session {active_session_id}"
                    )
                    need_force_takeover = True

            if mm.is_generating and active_session_id == session_id:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Generation is already running for this session.",
                    }
                ), 400

            # Set this session as active
            active_session_id = session_id

        # Clear previous session's consumption tracking if force takeover (no nested locks)
        if need_force_takeover:
            with consumption_monitor_lock:
                last_frame_consumed_time = None

        # Reset and start generation
        telemetry.reset()
        telemetry.log("server_session_start", session_id=session_id, text=text,
                      history_length=history_length)
        mm.reset(history_length=history_length, smoothing_alpha=smoothing_alpha)
        mm.start_generation(text, history_length=history_length)

        # Initialize consumption tracking (no nested locks)
        with consumption_monitor_lock:
            last_frame_consumed_time = time.time()

        # Start consumption monitoring
        start_consumption_monitor()
        print(f"[Session {session_id}] Consumption monitoring activated")

        return jsonify(
            {
                "status": "success",
                "message": f"Generation started with text: {text}, history_length: {history_length}",
                "session_id": session_id,
            }
        )
    except Exception as e:
        print(f"Error in start_generation: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/update_text", methods=["POST"])
def update_text():
    """Update the generation text"""
    try:
        data = request.json
        session_id = data.get("session_id")
        text = data.get("text", "")

        if not session_id:
            return jsonify(
                {"status": "error", "message": "session_id is required"}
            ), 400

        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify(
                    {"status": "error", "message": "Not the active session"}
                ), 403

        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400

        telemetry.log("server_update_request", session_id=session_id, text=text)
        model_manager.update_text(text)

        return jsonify({"status": "success", "message": f"Text updated to: {text}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/telemetry", methods=["POST"])
def client_telemetry():
    """Receive client-side telemetry events and append to the unified log."""
    try:
        data = request.json or {}
        src = data.pop("src", "client_event")
        telemetry.log(src, **data)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/pause", methods=["POST"])
def pause_generation():
    """Pause generation (keeps state for resume)"""
    try:
        data = request.json if request.json else {}
        session_id = data.get("session_id")

        if not session_id:
            return jsonify(
                {"status": "error", "message": "session_id is required"}
            ), 400

        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify(
                    {"status": "error", "message": "Not the active session"}
                ), 403

        if model_manager:
            model_manager.pause_generation()

        return jsonify({"status": "success", "message": "Generation paused"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/resume", methods=["POST"])
def resume_generation():
    """Resume generation from paused state"""
    try:
        global last_frame_consumed_time

        data = request.json if request.json else {}
        session_id = data.get("session_id")

        if not session_id:
            return jsonify(
                {"status": "error", "message": "session_id is required"}
            ), 400

        # Verify this is the active session
        with session_lock:
            if active_session_id != session_id:
                return jsonify(
                    {"status": "error", "message": "Not the active session"}
                ), 403

        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400

        model_manager.resume_generation()

        # Reset consumption tracking when resuming
        with consumption_monitor_lock:
            last_frame_consumed_time = time.time()

        return jsonify({"status": "success", "message": "Generation resumed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset generation state"""
    try:
        global active_session_id, last_frame_consumed_time

        data = request.json if request.json else {}
        session_id = data.get("session_id")
        history_length = data.get("history_length")
        smoothing_alpha = data.get("smoothing_alpha")

        # If session_id provided, verify it's the active session
        if session_id:
            with session_lock:
                if active_session_id and active_session_id != session_id:
                    return jsonify(
                        {"status": "error", "message": "Not the active session"}
                    ), 403

        if model_manager:
            model_manager.reset(
                history_length=history_length, smoothing_alpha=smoothing_alpha
            )

        # Clear the active session
        with session_lock:
            if active_session_id == session_id or not session_id:
                active_session_id = None

        # Clear consumption tracking
        with consumption_monitor_lock:
            last_frame_consumed_time = None

        print(f"[Session {session_id}] Reset complete, session cleared")

        return jsonify(
            {
                "status": "success",
                "message": "Reset complete",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/get_frame", methods=["GET"])
def get_frame():
    """Get the next frame"""
    try:
        global last_frame_consumed_time

        session_id = request.args.get("session_id")

        if not session_id:
            return jsonify(
                {"status": "error", "message": "session_id is required"}
            ), 400

        if model_manager is None:
            return jsonify({"status": "error", "message": "Model not initialized"}), 400

        count = min(int(request.args.get("count", 8)), 20)

        # Check if this is the active session or a spectator
        with session_lock:
            is_active = active_session_id == session_id

        if is_active:
            # Active session: pop frames from generation buffer
            frames = []
            for _ in range(count):
                joints = model_manager.get_next_frame()
                if joints is None:
                    break
                frames.append(joints.tolist())

            if frames:
                with consumption_monitor_lock:
                    last_frame_consumed_time = time.time()

                telemetry.log("server_frames_served", n=len(frames),
                              first_root=frames[0][0],
                              last_root=frames[-1][0],
                              buf_size=model_manager.frame_buffer.size())
                return jsonify(
                    {
                        "status": "success",
                        "frames": frames,
                        "buffer_size": model_manager.frame_buffer.size(),
                    }
                )
        else:
            # Spectator: read from broadcast buffer (non-destructive)
            after_id = int(request.args.get("after_id", 0))
            broadcast = model_manager.get_broadcast_frames(after_id, count)
            if broadcast:
                last_id = broadcast[-1][0]
                frames = [joints.tolist() for _, joints in broadcast]
                return jsonify(
                    {
                        "status": "success",
                        "frames": frames,
                        "last_id": last_id,
                        "buffer_size": model_manager.frame_buffer.size(),
                    }
                )

        # No frames available (active or spectator)
        return jsonify(
            {
                "status": "waiting",
                "message": "No frame available yet",
                "buffer_size": model_manager.frame_buffer.size(),
            }
        )
    except Exception as e:
        print(f"Error in get_frame: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get generation status"""
    try:
        session_id = request.args.get("session_id")

        with session_lock:
            is_active_session = session_id and active_session_id == session_id
            current_active_session = active_session_id

        if model_manager is None:
            return jsonify(
                {
                    "initialized": False,
                    "buffer_size": 0,
                    "is_generating": False,
                    "is_active_session": is_active_session,
                    "active_session_id": current_active_session,
                }
            )

        status = model_manager.get_buffer_status()
        status["initialized"] = True
        status["is_active_session"] = is_active_session
        status["active_session_id"] = current_active_session

        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask server for real-time 3D motion generation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ShandaAI/FloodDiffusionTiny",
        help="HF Hub model name (default: ShandaAI/FloodDiffusionTiny)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    args = parser.parse_args()

    model_name_global = args.model_name

    # Load model eagerly on startup (pre-downloaded in Docker)
    print(f"Loading model: {model_name_global}")
    init_model()

    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
