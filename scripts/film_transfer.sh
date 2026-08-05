#!/bin/bash
# Film one transfer clip: dialog gate, then fullscreen looping playback.
# Usage: ./film_transfer.sh [video]   (default: demo/transmit_mono.mp4)
set -e
cd "$(dirname "$0")/.."
VIDEO="${1:-demo/transmit_mono.mp4}"

osascript >/dev/null <<EOF
display dialog "Transfer capture — $(basename "$VIDEO")

Phone: normal VIDEO mode, landscape, 1x lens.
Position: ~40 cm, code filling the frame, held steady.
Long-press the viewfinder for AE/AF LOCK, then park the mouse in a corner.

Click Start. 8 seconds to frame up, then record ~25 seconds and stop.
AirDrop the clip after." buttons {"Start"} default button 1
EOF
sleep 8
ffplay -fs -loop 4 -autoexit "$VIDEO" >/dev/null 2>&1 || true
