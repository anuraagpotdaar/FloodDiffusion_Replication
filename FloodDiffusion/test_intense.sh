#!/usr/bin/env bash
# Intense end-to-end test. Cycles through every chip motion N times back-to-back
# at the new "burst" cadence (one quota's worth = ~3 s) and reports per-prompt
# motion stats AGGREGATED ACROSS ALL ITERATIONS so we can verify each prompt
# produces a consistent motion every time.

set -euo pipefail
PORT="${PORT:-5050}"
SESSION="intense-test-$$"
ITERATIONS="${ITERATIONS:-3}"
HOLD_S="${HOLD_S:-3.5}"   # slightly longer than the 60-frame quota at 20 FPS

PROMPTS=(
  "walk forward" "run forward" "jump" "dance" "sit down" "stand up"
  "wave hand" "kick" "punch" "crouch" "turn around" "clap hands"
  "bow" "walk in a circle"
)

step() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

step "Start session '$SESSION' with text='${PROMPTS[0]}'"
curl -sS -X POST "http://localhost:${PORT}/api/start" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s","text":"%s","force":true}' "$SESSION" "${PROMPTS[0]}")" \
  | tee /dev/stderr >/dev/null

# Background frame consumer keeps the session alive (otherwise consumption monitor
# would eventually free the session and update_text would 403).
(
    while :; do
        curl -sS "http://localhost:${PORT}/api/get_frame?session_id=${SESSION}&count=2" \
            >/dev/null 2>&1 || break
        sleep 0.1
    done
) &
CONSUMER_PID=$!
trap "kill $CONSUMER_PID 2>/dev/null || true" EXIT

sleep "$HOLD_S"
for ((it=1; it<=ITERATIONS; it++)); do
    for ((i=0; i<${#PROMPTS[@]}; i++)); do
        # On iteration 1 we already played PROMPTS[0] via /api/start.
        if [[ $it -eq 1 && $i -eq 0 ]]; then continue; fi
        step "iter=$it [${i}/${#PROMPTS[@]}] -> '${PROMPTS[$i]}'"
        curl -sS -X POST "http://localhost:${PORT}/api/update_text" \
             -H 'Content-Type: application/json' \
             -d "$(printf '{"session_id":"%s","text":"%s"}' "$SESSION" "${PROMPTS[$i]}")" >/dev/null
        sleep "$HOLD_S"
    done
done

step "Pause"
curl -sS -X POST "http://localhost:${PORT}/api/pause" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s"}' "$SESSION")" >/dev/null

step "Aggregated per-prompt motion stats"
python3 - <<'PY'
import json, statistics
events = [json.loads(l) for l in open('telemetry.jsonl') if l.strip()]
t0 = events[0]['t']

# Group frames by their conditioning text
by_text = {}
for e in events:
    if e['src'] == 'frame_added':
        by_text.setdefault(e['text'], []).append(e['root'])

# Find the latency between each update_text_done and the first frame_added under it
pending_resp = []
last_update_per_text = {}
for e in events:
    if e['src'] == 'update_text_done':
        last_update_per_text[e['new']] = e['t']
for text, ts in last_update_per_text.items():
    first_after = next((e for e in events
                        if e['src']=='frame_added' and e.get('text')==text and e['t']>=ts),
                       None)
    if first_after is not None:
        pending_resp.append(first_after['t'] - ts)

done_events = [e for e in events if e['src']=='action_done']

print(f"Iterations:           {sum(1 for e in events if e['src']=='update_text_done')+1} text-blocks")
print(f"Action completions:   {len(done_events)} (each action stops cleanly)")
reasons = {}
for e in done_events:
    r = e.get('reason', 'unknown').split()[0]
    reasons[r] = reasons.get(r, 0) + 1
print(f"  by reason: {reasons}")
if pending_resp:
    print(f"Update -> first frame: median {statistics.median(pending_resp)*1000:.0f}ms, "
          f"max {max(pending_resp)*1000:.0f}ms, n={len(pending_resp)}")

print()
print(f"{'prompt':<20} {'n':>5} {'hipY_mu':>9} {'hipY_sd':>9} {'X_drift':>8} {'Z_drift':>8}")
for text in sorted(by_text):
    fr = by_text[text]
    if len(fr) < 2: continue
    ys = [r[1] for r in fr]
    xs = [r[0] for r in fr]
    zs = [r[2] for r in fr]
    print(f"{text:<20} {len(fr):>5} "
          f"{sum(ys)/len(ys):>9.3f} {statistics.stdev(ys):>9.4f} "
          f"{xs[-1]-xs[0]:>+8.2f} {zs[-1]-zs[0]:>+8.2f}")

# Sanity: did frame counts stay roughly proportional to the quota across iterations?
# (i.e. does one update produce ~one quota's worth of frames?)
print()
quota_default = 60
spans = {}  # text -> list of frames-produced-per-update_text
ordered_updates = [e for e in events if e['src']=='update_text_done']
for i, upd in enumerate(ordered_updates):
    end_t = ordered_updates[i+1]['t'] if i+1 < len(ordered_updates) else float('inf')
    n = sum(1 for e in events
            if e['src']=='frame_added' and e.get('text')==upd['new']
            and upd['t'] <= e['t'] < end_t)
    spans.setdefault(upd['new'], []).append(n)
print(f"Frames per action (min={40}, max={240}):")
for text in sorted(spans):
    runs = spans[text]
    print(f"  {text:<20} n_runs={len(runs)} frames per run = {runs}")
PY
