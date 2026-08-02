#!/bin/bash
# RECORD TAKE, 252x163. The grid now fills the panel: 3024x1956 of 3024x1964,
# at the same 12px cells (so the same ~13.4 camera-px/cell that decodes) as the
# old 252x140. That is 19 codewords per frame instead of 16 - a 226.0 KB/s
# ceiling instead of 190.3, which is what makes 200 KB/s reachable at all.
#
# There is NO separate countdown clip any more. The old one was rendered from a
# different transmit, so it carried a perfectly valid header for a different
# file, and "first header wins" learned k=5525/1.12MB from it and then tried to
# rebuild that from a 277KB transmission. The lead-in here is the transmit
# itself: luminance-matched by construction, and every frame of it is real data.
cd "$(dirname "$0")"
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to true' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to true' 2>/dev/null
sleep 1
cat <<'EOF'
------------------------------------------------------------------
  1. Evening light, ONE lamp. Not daylight. The 12 KB/s take was
     filmed at 12:55 in daylight and read p50=180 with the blacks
     lifted; the 110 KB/s take was filmed in the evening.
  2. Stock Camera, 4K 60, landscape, 1x. Fill the frame with the
     screen.
  3. Hit record, then tap-and-hold to LOCK AE/AF immediately.
  4. Do NOT touch the exposure slider. It cuts exposure without
     raising ISO, so a short shutter necessarily underexposes - a
     take at 3 stops down read full-white at 25/255. Straddle is
     fought with display hold, not exposure.
  5. Then HOLD STILL and do nothing for the rest of the take. The
     camera needs ~7 seconds to settle after the lock (measured:
     BER 8.23% -> 1.51%, codewords 0 -> 12/16, monotonic). The
     first seconds are expected to be poor and cost nothing - the
     decoder reports the best window as well as the full span.
------------------------------------------------------------------
EOF
sleep 3
echo "22s LOCK-IN with an on-screen countdown, then ~75s of code."
T0=$(python3 -c 'import time;print(time.time())')
# The lock-in is this transmit's OWN frames with the countdown drawn over
# them: same header, same file, so it cannot teach the receiver the wrong k
# the way the old shared countdown clip did. Separate file so the countdown
# plays once rather than every time the code loop wraps.
ffplay -v error -fs -alwaysontop -noborder -autoexit demo/_tx_record163_lead.mp4
ffplay -v error -fs -alwaysontop -noborder -loop 5 -autoexit demo/_tx_record163.mp4
T1=$(python3 -c 'import time;print(time.time())')
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to false' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false' 2>/dev/null
python3 -c "print(f'displayed {$T1-$T0:.1f}s')"
cat <<'EOF'
stop recording, AirDrop it, then:

  python3 poc/fast_decode.py ~/Downloads/IMG_XXXX.MOV /tmp/out.bin \
      --grid 252x163 --ecc 48 --subblock --soft --scan
EOF
