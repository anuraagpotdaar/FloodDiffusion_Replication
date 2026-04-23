#!/usr/bin/env bash
# Multi-prompt switching test. Cycles through a few motions and reports per-motion
# motion statistics so we can confirm the model actually responds differently.

set -euo pipefail
PORT="${PORT:-5050}"
SESSION="multi-test-$$"
PROMPTS=("walk forward" "jump" "sit down" "spin around" "wave hand")
DURATION_PER_PROMPT=6

step() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

step "Start with first prompt: ${PROMPTS[0]}"
curl -sS -X POST "http://localhost:${PORT}/api/start" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s","text":"%s","force":true}' "$SESSION" "${PROMPTS[0]}")" \
  | tee /dev/stderr >/dev/null

(
    while :; do
        curl -sS "http://localhost:${PORT}/api/get_frame?session_id=${SESSION}&count=2" \
            >/dev/null 2>&1 || break
        sleep 0.1
    done
) &
CONSUMER_PID=$!
trap "kill $CONSUMER_PID 2>/dev/null || true" EXIT

sleep $DURATION_PER_PROMPT
for ((i=1; i<${#PROMPTS[@]}; i++)); do
    step "Switch to: ${PROMPTS[$i]}"
    curl -sS -X POST "http://localhost:${PORT}/api/update_text" \
         -H 'Content-Type: application/json' \
         -d "$(printf '{"session_id":"%s","text":"%s"}' "$SESSION" "${PROMPTS[$i]}")" >/dev/null
    sleep $DURATION_PER_PROMPT
done

step "Pause"
curl -sS -X POST "http://localhost:${PORT}/api/pause" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s"}' "$SESSION")" >/dev/null

step "Per-prompt motion analysis"
python3 - <<'PY'
import json, statistics
events = [json.loads(l) for l in open('telemetry.jsonl') if l.strip()]
t0 = events[0]['t']
by_text = {}
for e in events:
    if e['src'] == 'frame_added':
        by_text.setdefault(e['text'], []).append((e['t']-t0, e['root']))

print(f"{'prompt':<20} {'n':>4} {'hipY_mean':>10} {'hipY_std':>10} {'X_drift':>8} {'Z_drift':>8} {'response_ms':>11}")
upd_times = {e['new']: e['t']-t0 for e in events if e['src']=='update_text_done'}
first_step_after = {}
for e in events:
    if e['src']=='gen_step_start' and e['text'] in upd_times and e['text'] not in first_step_after:
        if e['t']-t0 >= upd_times[e['text']]:
            first_step_after[e['text']] = e['t']-t0

for text, fr in by_text.items():
    ys = [r[1] for _, r in fr]
    xs = [r[0] for _, r in fr]
    zs = [r[2] for _, r in fr]
    if len(ys) < 2: continue
    response = ""
    if text in upd_times and text in first_step_after:
        response = f"{(first_step_after[text]-upd_times[text])*1000:.0f}"
    print(f"{text:<20} {len(ys):>4} {sum(ys)/len(ys):>10.3f} {statistics.stdev(ys):>10.4f} "
          f"{xs[-1]-xs[0]:>+8.2f} {zs[-1]-zs[0]:>+8.2f} {response:>11}")
PY
