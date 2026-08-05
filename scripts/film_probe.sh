#!/bin/bash
# One take, whole operating envelope. Hides Dock/menu bar, plays a
# luminance-matched countdown so an AE/AF lock carries over, then the probe.
cd "$(dirname "$0")/.."
D=demo
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to true' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to true' 2>/dev/null
sleep 1
echo "countdown x2 (~24s): hit record + tap-hold to lock EARLY, then hold still."
echo "the camera needs ~7s to settle after the lock - measured, BER 8.2% -> 1.5%."
ffplay -v error -fs -alwaysontop -noborder -loop 2 -autoexit "$D/_probe_countdown.mp4"
echo "probe playing, 4 loops (~57s). keep filming until the screen goes normal."
T0=$(python3 -c 'import time;print(time.time())')
ffplay -v error -fs -alwaysontop -noborder -loop 4 -autoexit "$D/_tx_probe_fs3024x1964.mp4"
T1=$(python3 -c 'import time;print(time.time())')
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to false' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false' 2>/dev/null
python3 -c "print(f'displayed for {$T1-$T0:.1f}s')"
echo "stop recording, AirDrop it over."
