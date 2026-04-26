#!/usr/bin/env bash
# Chip-by-chip motion benchmark. Runs each chip prompt in isolation with a fresh
# session, captures the resulting motion, and compares against expected per-chip
# characteristics (hip-Y range, drift, in-place vs translation, etc.).
#
# Each prompt gets ~6 s of generation. For one-shots that's enough for the
# action to complete + we observe its full arc. For continuous motions, 6 s is
# representative of steady-state behaviour.

set -euo pipefail
PORT="${PORT:-5050}"
HOLD_S="${HOLD_S:-6}"
TELEM=/Users/anuraag/Developer/masters/ccn/FloodDiffusion_Replication/FloodDiffusion/telemetry.jsonl
PER_PROMPT_LOG=/tmp/chip_benchmark_data.jsonl

# (prompt, expected) — expected is a Python literal dict with the assertions
# we'll check. Values are loose bounds (the model is a tiny streaming demo,
# not an Olympic motion-capture rig) — we want to flag obvious drift, not
# nitpick exact numbers.
PROMPTS=(
  "walk forward"
  "run forward"
  "jump"
  "dance"
  "sit down"
  "stand up"
  "wave hand"
  "kick"
  "punch"
  "crouch"
  "turn around"
  "clap hands"
  "bow"
  "walk in a circle"
)

step() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

: > "$PER_PROMPT_LOG"

for prompt in "${PROMPTS[@]}"; do
    SESSION="bench-$$-$(echo "$prompt" | tr -dc 'a-z')"
    step "→ '$prompt' (session=$SESSION)"

    curl -sS -X POST "http://localhost:${PORT}/api/start" \
         -H 'Content-Type: application/json' \
         -d "$(printf '{"session_id":"%s","text":"%s","force":true}' "$SESSION" "$prompt")" \
        >/dev/null

    ( while :; do
        curl -sS "http://localhost:${PORT}/api/get_frame?session_id=${SESSION}&count=2" \
            >/dev/null 2>&1 || break
        sleep 0.1
      done ) &
    CONS=$!

    sleep "$HOLD_S"

    # Snapshot the relevant slice of telemetry for this prompt and append to log
    python3 - "$prompt" "$PER_PROMPT_LOG" "$TELEM" <<'PY'
import json, sys
prompt, out, telem = sys.argv[1:4]
events = []
last_session_start = None
with open(telem) as f:
    for line in f:
        try: e = json.loads(line)
        except: continue
        if e.get('src') == 'server_session_start':
            last_session_start = e
            events = [e]
        else:
            events.append(e)
out_events = [e for e in events
               if e.get('src') == 'frame_added' and e.get('text') == prompt]
with open(out, 'a') as f:
    f.write(json.dumps({'prompt': prompt, 'frames': len(out_events),
                        'samples': [e['root'] for e in out_events]}) + '\n')
PY

    curl -sS -X POST "http://localhost:${PORT}/api/pause" \
         -H 'Content-Type: application/json' \
         -d "$(printf '{"session_id":"%s"}' "$SESSION")" >/dev/null
    kill $CONS 2>/dev/null || true
    sleep 0.5
done

step "Per-chip benchmark report"
python3 - "$PER_PROMPT_LOG" <<'PY'
import json, statistics, sys
log_path = sys.argv[1]

# Expected ranges per prompt. Conventions:
#   hipY_mu: rough mean hip height (m). Standing ~0.95, sitting ~0.4-0.6, crouching ~0.3-0.5.
#   hipY_sd_min/max: stdev of hip Y across the sample. High = bouncy/jumpy.
#   xz_drift_max: largest acceptable horizontal travel for "in place" motions (m).
#   in_place: True if the character should mostly stay put.
EXPECT = {
    "walk forward":      {"hipY_mu": (0.85, 1.00), "hipY_sd": (0.005, 0.07), "in_place": False},
    "run forward":       {"hipY_mu": (0.85, 1.00), "hipY_sd": (0.010, 0.10), "in_place": False},
    "jump":              {"hipY_mu": (0.85, 1.20), "hipY_sd": (0.030, 0.30), "in_place": True},
    "dance":             {"hipY_mu": (0.80, 1.05), "hipY_sd": (0.005, 0.20), "in_place": False},
    "sit down":          {"hipY_mu": (0.20, 0.85), "hipY_sd": (0.020, 0.40), "in_place": True},
    "stand up":          {"hipY_mu": (0.30, 1.00), "hipY_sd": (0.020, 0.40), "in_place": True},
    "wave hand":         {"hipY_mu": (0.80, 1.05), "hipY_sd": (0.000, 0.15), "in_place": True},
    "kick":              {"hipY_mu": (0.80, 1.20), "hipY_sd": (0.005, 0.20), "in_place": True},
    "punch":             {"hipY_mu": (0.80, 1.05), "hipY_sd": (0.000, 0.10), "in_place": True},
    "crouch":            {"hipY_mu": (0.20, 0.85), "hipY_sd": (0.010, 0.30), "in_place": True},
    "turn around":       {"hipY_mu": (0.85, 1.00), "hipY_sd": (0.000, 0.10), "in_place": True},
    "clap hands":        {"hipY_mu": (0.85, 1.05), "hipY_sd": (0.000, 0.05), "in_place": True},
    "bow":               {"hipY_mu": (0.80, 1.00), "hipY_sd": (0.000, 0.15), "in_place": True},
    "walk in a circle":  {"hipY_mu": (0.85, 1.00), "hipY_sd": (0.005, 0.07), "in_place": False},
}
IN_PLACE_DRIFT_MAX = 2.0  # metres of horizontal travel allowed for "in place" prompts

def fmt(v, ok):
    return f"\033[32m{v}\033[0m" if ok else f"\033[31m{v}\033[0m"

print(f"{'prompt':<20} {'frames':>6}  {'hipY_mu':>8}  {'hipY_sd':>8}  "
      f"{'X_drift':>8}  {'Z_drift':>8}  {'in_place':>8}  notes")
print('-' * 100)

passes = fails = 0
for line in open(log_path):
    rec = json.loads(line)
    p = rec['prompt']; samples = rec['samples']
    if not samples:
        print(f"{p:<20} {'NO DATA':>6}  -        -        -         -         -        no frames captured")
        fails += 1; continue
    ys = [s[1] for s in samples]
    xs = [s[0] for s in samples]
    zs = [s[2] for s in samples]
    mu = sum(ys)/len(ys)
    sd = statistics.stdev(ys) if len(ys) > 1 else 0.0
    xdrift = xs[-1] - xs[0]
    zdrift = zs[-1] - zs[0]
    horiz = (xdrift**2 + zdrift**2) ** 0.5

    exp = EXPECT.get(p, {})
    notes = []
    ok_mu = True; ok_sd = True; ok_inplace = True
    if 'hipY_mu' in exp:
        lo, hi = exp['hipY_mu']
        ok_mu = lo <= mu <= hi
        if not ok_mu: notes.append(f"hipY_mu {mu:.3f} outside [{lo},{hi}]")
    if 'hipY_sd' in exp:
        lo, hi = exp['hipY_sd']
        ok_sd = lo <= sd <= hi
        if not ok_sd: notes.append(f"hipY_sd {sd:.3f} outside [{lo},{hi}]")
    if exp.get('in_place'):
        ok_inplace = horiz <= IN_PLACE_DRIFT_MAX
        if not ok_inplace: notes.append(f"travelled {horiz:.2f}m (expected in-place)")

    print(f"{p:<20} {len(samples):>6}  "
          f"{fmt(f'{mu:>8.3f}', ok_mu)}  "
          f"{fmt(f'{sd:>8.4f}', ok_sd)}  "
          f"{xdrift:>8.2f}  "
          f"{zdrift:>8.2f}  "
          f"{fmt(f'{str(horiz <= IN_PLACE_DRIFT_MAX):>8}', ok_inplace if exp.get('in_place') else True)}  "
          f"{'; '.join(notes) if notes else 'OK'}")
    if all([ok_mu, ok_sd, ok_inplace]): passes += 1
    else: fails += 1

print('-' * 100)
print(f"Summary: {passes} pass, {fails} fail")
PY
