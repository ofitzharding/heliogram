#!/bin/bash
# LUMINANCE LADDER. IMG_7879 did not test gray4 - it tested a saturated sensor.
# Mono, the control on that same take, fell to 23.6% from the ~90% this rig
# gives when the exposure is right, and the eye read p5=32 / p50=189 against a
# midpoint of 135. gray4's two middle levels are the first casualty of a
# compressed transfer curve, so it read 0% without ever being measured.
#
# The phone's exposure is not adjustable from here (AE is locked by you, and
# the slider makes it worse). But the LIGHT LEVEL is a transmitter property,
# so this take sweeps it: gray4 at peak white 255, 200, 160 and 120, with a
# mono section at full peak in every loop as the control.
#
# Costs nothing on the receiver - the gray4 demodulator learns its four levels
# by k-means from each frame, so it adapts to whatever peak arrives.
cd "$(dirname "$0")/.."
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to true' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to true' 2>/dev/null
sleep 1
cat <<'EOF'
------------------------------------------------------------------
  Same discipline as before:
    Evening light, ONE lamp. Stock Camera, 4K 60, landscape, 1x.
    Fill the frame with the screen. Hit record, tap-and-hold to
    LOCK AE/AF during the countdown, do NOT touch the exposure
    slider, then hold still.

  NEW, and it matters for this take: put the SCREEN BRIGHTNESS
  somewhere in the middle, not maximum. Screen brightness and
  transmit peak multiply. The ladder sweeps a 2.1x range (255 ->
  120); if the panel is at full brightness in a dark room, even
  the dimmest rung can still saturate and the whole ladder comes
  back flat at 0%.

  The take is ~112s: 22s countdown, then 3 loops x 30s.
------------------------------------------------------------------
EOF
sleep 5
T0=$(python3 -c 'import time;print(time.time())')
ffplay -v error -fs -alwaysontop -noborder -autoexit demo/_tx_lumenprobe_lead.mp4
ffplay -v error -fs -alwaysontop -noborder -loop 3 -autoexit demo/_tx_lumenprobe.mp4
T1=$(python3 -c 'import time;print(time.time())')
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to false' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false' 2>/dev/null
python3 -c "print(f'displayed {$T1-$T0:.1f}s')"
cat <<'EOF'
stop recording, AirDrop it, then:

  python3 src/analyze_lumen.py ~/Downloads/IMG_XXXX.MOV

Read the MONO row first. If it is not back near ~90%, the take is bad and no
gray4 row means anything. If mono is healthy, read gray4 down the peak column
and compare KB/s - not yield. gray4 carries twice the bits, so it only has to
be half as good as mono to win.
EOF
