#!/bin/bash
# Guided capture session: the Mac drives, the human only films.
# Phase 1: smeartest --auto (alternate 25s -> split 25s -> static 25s), one slo-mo clip.
# Phase 2: fullscreen transfer video, one normal-video clip.
set -e
cd "$(dirname "$0")"

osascript >/dev/null <<'EOF'
display dialog "PHASE 1 of 2 — Smear test (~75 seconds)

Phone: Camera app → SLO-MO. Check Settings → Camera → Record Slo-mo is 1080p at 240 fps.
Room: as dim as you can make it. Screen stays bright.
Position: ~50 cm away, screen filling the frame, square-on.
Long-press the phone screen on the display until AE/AF LOCK appears.

Click Start. You then have 8 seconds to raise the phone and START RECORDING.
Record ONE continuous clip until the flashing stops on its own (~75 s), then stop recording." buttons {"Start"} default button 1
EOF
sleep 8
./experiments/smear-test/smeartest --auto || true

osascript >/dev/null <<'EOF'
display dialog "PHASE 2 of 2 — Real file transfer (~45 seconds)

Phone: switch to normal VIDEO mode (not slo-mo).
Position: ~40 cm, the colored code filling the frame, held as steady as you can.

Click Start. 8 seconds to frame up, then record ~20 seconds and stop." buttons {"Start"} default button 1
EOF
sleep 8
ffplay -fs -loop 6 -autoexit demo/transmit.mp4 >/dev/null 2>&1 || true

osascript >/dev/null <<'EOF'
display dialog "Done. AirDrop BOTH clips to this Mac, then tell Claude: clips sent." buttons {"OK"} default button 1
EOF
