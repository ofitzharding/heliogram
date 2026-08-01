#!/bin/bash
# Manual playback. Nothing is timed, nothing runs in the background.
# You start it, you stop it (press q or Esc in the video window).
cd "$(dirname "$0")"
V="${1:-demo/tx_dense_big.mp4}"
echo "Playing $V fullscreen. Press q to stop."
ffplay -fs -alwaysontop -noborder -loop 15 -autoexit "$V"
