#!/bin/bash
# GRAY4 PROBE. One take decides whether 2 bits/cell doubles the link.
#
#   252x163 mono  ecc 48   19 codewords/frame   226.0 KB/s   200 needs 88.5%
#   252x163 gray4 ecc 48   38 codewords/frame   452.0 KB/s   200 needs 44.2%
#   252x163 gray4 ecc 64   38 codewords/frame   416.4 KB/s   200 needs 48.0%
#
# gray4 does not have to be as good as mono to win - it has to be HALF as
# good. The three configurations are interleaved in one video at a fixed grid
# and cell size, so optics, framing, exposure and hold are pinned across the
# comparison and the only thing varying is the alphabet and the parity. Each
# frame states its own mode and ecc in its header, so the analyser attributes
# it by content, not by frame offset.
#
# ecc 64 is the hedge: if gray4/48 fails and gray4/64 holds, the alphabet is
# fine and the parity was the binding constraint.
cd "$(dirname "$0")/.."
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to true' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to true' 2>/dev/null
sleep 1
cat <<'EOF'
------------------------------------------------------------------
  Evening light, ONE lamp. Not daylight.
  Stock Camera, 4K 60, landscape, 1x. Fill the frame with the screen.
  Hit record, then tap-and-hold to LOCK AE/AF during the countdown.
  Do NOT touch the exposure slider.
  Then HOLD STILL for the whole take.

  gray4 has 4 luminance levels instead of 2, so its decision
  margins are ~1/3 as wide as mono's. Exposure discipline matters
  more here than on any previous take: a lifted black or a
  compressed white end collapses the middle two levels and the
  answer comes back falsely negative.
------------------------------------------------------------------
EOF
sleep 4
echo "22s lock-in countdown, then 3 loops x 24s of probe (~94s total)."
T0=$(python3 -c 'import time;print(time.time())')
ffplay -v error -fs -alwaysontop -noborder -autoexit demo/_tx_grayprobe_lead.mp4
ffplay -v error -fs -alwaysontop -noborder -loop 3 -autoexit demo/_tx_grayprobe.mp4
T1=$(python3 -c 'import time;print(time.time())')
osascript -e 'tell application "System Events" to tell dock preferences to set autohide to false' \
          -e 'tell application "System Events" to tell dock preferences to set autohide menu bar to false' 2>/dev/null
python3 -c "print(f'displayed {$T1-$T0:.1f}s')"
cat <<'EOF'
stop recording, AirDrop it, then:

  python3 src/analyze_gray.py ~/Downloads/IMG_XXXX.MOV

Read the result as: gray4 wins if its KB/s column beats mono's, even at a
much lower yield percentage. Yield is not the figure of merit here, KB/s is.
EOF
