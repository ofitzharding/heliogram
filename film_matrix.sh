#!/bin/bash
# The honest 2x2: {lights on, dark} x {handheld, propped}, one video, four clips.
# Same Blackmagic settings for all four: 4K, 60fps (or 120 if available),
# shutter 1/500, ISO 400, WB + focus locked, 1x lens, ~45cm, code ~2/3 of frame.
# Name each clip in AirDrop order; the decode step maps them by timestamp.
set -e
cd "$(dirname "$0")"
VIDEO="${1:-demo/transmit_c4.mp4}"

run_condition() {
  osascript >/dev/null <<EOF
display dialog "Condition $1 of 4: $2

$3

Click Start, 8s to frame, record ~25s, stop." buttons {"Start"} default button 1
EOF
  sleep 8
  ffplay -fs -loop 4 -autoexit "$VIDEO" >/dev/null 2>&1 || true
}

run_condition 1 "LIGHTS ON + HANDHELD" \
  "Room lights ON (normal daytime room). Hold the phone in your hands. This is the headline number: the realistic case."
run_condition 2 "LIGHTS ON + PROPPED" \
  "Lights stay ON. Prop the phone on books/stand, dead-on, then don't touch it."
run_condition 3 "DARK + HANDHELD" \
  "Lights OFF, curtains shut. Back to holding the phone."
run_condition 4 "DARK + PROPPED" \
  "Lights stay OFF. Prop the phone again. This is the best-case reference."

osascript >/dev/null <<'EOF'
display dialog "Done. AirDrop all FOUR clips, then tell Claude: matrix filmed.
They are matched to conditions by file timestamp order (1->4)." buttons {"OK"} default button 1
EOF
