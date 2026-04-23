#!/usr/bin/env bash
# End-to-end test of the text-update path. Bypasses the browser entirely.
#
#   ./test_text_switch.sh
#
# 1. Starts a fresh session with a known prompt
# 2. Lets it generate for ~6 s (enough for several chunks)
# 3. Switches the prompt
# 4. Lets it generate for ~10 s under the new prompt
# 5. Stops
# Then prints a timeline-summary of telemetry.jsonl.

set -euo pipefail

PORT="${PORT:-5050}"
SESSION="cli-test-$$"
STAGE1_TEXT="walk forward"
STAGE2_TEXT="jump"

step() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

step "Start session '$SESSION' with text='$STAGE1_TEXT'"
curl -sS -X POST "http://localhost:${PORT}/api/start" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s","text":"%s","force":true}' "$SESSION" "$STAGE1_TEXT")" \
  | tee /dev/stderr >/dev/null

# Consume frames in the background to keep the session "active" (otherwise the
# consumption monitor will auto-reset us and update_text will return 403).
(
    while :; do
        curl -sS "http://localhost:${PORT}/api/get_frame?session_id=${SESSION}&count=2" \
            >/dev/null 2>&1 || break
        sleep 0.1
    done
) &
CONSUMER_PID=$!
trap "kill $CONSUMER_PID 2>/dev/null || true" EXIT

step "Drain frames so the buffer is realistic (6 s)"
sleep 6

step "Update text to '$STAGE2_TEXT'"
curl -sS -X POST "http://localhost:${PORT}/api/update_text" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s","text":"%s"}' "$SESSION" "$STAGE2_TEXT")" \
  | tee /dev/stderr >/dev/null

step "Generate under new text (10 s)"
sleep 10

step "Pause"
curl -sS -X POST "http://localhost:${PORT}/api/pause" \
     -H 'Content-Type: application/json' \
     -d "$(printf '{"session_id":"%s"}' "$SESSION")" >/dev/null

step "Telemetry summary"
python3 - <<'PY'
import json, os, sys
path = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'telemetry.jsonl')
events = []
with open(path) as f:
    for line in f:
        try: events.append(json.loads(line))
        except: pass
if not events:
    print("(no events)"); sys.exit(0)
t0 = events[0]['t']
def fmt(e):
    rel = e['t'] - t0
    extras = {k: v for k, v in e.items() if k not in ('t','src')}
    return f"  {rel:6.2f}s  {e['src']:<26}  {extras}"
counts = {}
for e in events: counts[e['src']] = counts.get(e['src'], 0) + 1
print("counts:", counts)
print("\nfirst 80 events:")
for e in events[:80]: print(fmt(e))
print("\nlast 30 events:")
for e in events[-30:]: print(fmt(e))
print("\nupdate-related events:")
for e in events:
    if 'update' in e['src'] or 'fetch' in e['src']:
        print(fmt(e))
PY
