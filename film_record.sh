#!/bin/bash
# RECORD ATTEMPT. 252x140 mono - the density measured at 176-193 KB/s in the
# settled portion of the probe take. Long countdown because the camera's
# AF/AE transient is ~7s (measured: BER 8.23% -> 1.51% over frames 127-446).
cd "$(dirname "$0")"
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to true' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to true' 2>/dev/null
sleep 1
echo "COUNTDOWN (~24s). Hit record and tap-hold to LOCK immediately, then hold"
echo "still and let the camera settle. Do NOT touch the exposure slider."
ffplay -v error -fs -alwaysontop -noborder -loop 2 -autoexit demo/_probe_countdown.mp4
echo "transmitting - 4 loops, ~53s. keep filming until the screen goes normal."
T0=$(python3 -c 'import time;print(time.time())')
ffplay -v error -fs -alwaysontop -noborder -loop 4 -autoexit demo/_tx_record_fs.mp4
T1=$(python3 -c 'import time;print(time.time())')
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to false' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false' 2>/dev/null
python3 -c "print(f'displayed {$T1-$T0:.1f}s')"
echo "stop recording, AirDrop it."
