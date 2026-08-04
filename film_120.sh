#!/bin/bash
# 120 fps TAKE. The panel is ProMotion and reports 1512x982 @ 120.00Hz, but
# every transmit in this project has been rendered at 60 fps - so half the
# channel has never been used. the reference tool reports its 128 KB/s headline
# figure comes from "a 120 fps ProMotion sender"; we have been comparing a
# half-rate link against a full-rate one.
#
#   252x163 mono ecc48 @  60 fps   226.0 KB/s ceiling   200 needs 88.5% yield
#   252x163 mono ecc48 @ 120 fps   452.0 KB/s ceiling   200 needs 44.2% yield
#
# Same grid, same cell size, same alphabet, same decision margins - the only
# change is that the panel now shows a new frame every refresh instead of every
# other refresh. Unlike gray4 this costs no margin at all.
#
# THE CATCH, stated up front: 120 fps halves the exposure per camera frame, so
# each frame collects about half the photons. On a shot-noise-limited sensor
# that costs roughly 1.4x in decision margin (d'), and mono measured d'=3.7-5.6
# at 60 fps. If the yield loss is worse than the 2x frame gain, this is a wash.
# That is exactly what this take measures.
#
# RECORD AT 4K120. The iPhone 17 Pro Max does it. If the camera runs at 60 while
# the display runs at 120, every camera frame integrates TWO different code
# frames and nothing will decode at all.
cd "$(dirname "$0")"
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to true' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to true' 2>/dev/null
sleep 1
cat <<'EOF'
------------------------------------------------------------------
  CAMERA: 4K at 120 fps. This is the one setting that matters most.
          Settings > Camera > Record Video > 4K/120. If it records
          at 60, the take is dead on arrival - the display is
          showing 120 distinct frames a second.

  Screen brightness: HIGH this time, not mid. 120 fps halves the
  exposure per frame, so the sensor needs the light back. (The
  opposite of the gray4 advice, for the opposite reason.)

  Otherwise the same: evening light, one lamp, landscape, 1x,
  fill the frame, lock AE/AF during the countdown, do NOT touch
  the exposure slider, hold still.

  ~97s: 22s countdown, then 5 loops x 15s.
------------------------------------------------------------------
EOF
sleep 5
T0=$(python3 -c 'import time;print(time.time())')
ffplay -v error -fs -alwaysontop -noborder -autoexit demo/_tx_120_lead.mp4
ffplay -v error -fs -alwaysontop -noborder -loop 5 -autoexit demo/_tx_120.mp4
T1=$(python3 -c 'import time;print(time.time())')
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to false' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false' 2>/dev/null
python3 -c "
el=$T1-$T0
print(f'displayed {el:.1f}s')
exp=22.0+5*15.0
print(f'expected  {exp:.1f}s if the player sustained 120 fps')
if el > exp*1.25:
    print('WARNING: playback ran LONG, so frames were not presented at 120 fps.')
    print('The take will still decode, but at an effective rate below 120 -')
    print('check the fps the decoder reports against the capture.')
"
cat <<'EOF'
stop recording, AirDrop it, then:

  python3 poc/fast_decode.py ~/Downloads/IMG_XXXX.MOV /tmp/out.bin \
      --grid 252x163 --ecc 48 --subblock --soft --scan --full

Read GOODPUT (full span), not BEST WINDOW. Full span is the honest
time-to-file number and the only one comparable to the reference tool's.
EOF
