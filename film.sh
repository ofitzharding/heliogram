#!/bin/bash
# One session, three takes:  ./film.sh density | strobe | 120
#
#   density  ffplay video, camera 4K60.  Finds where the density cliff is.
#   strobe   browser, camera 4K60, screen brightness MAX. Display as shutter.
#   120      browser, camera 4K120.     Full-rate channel.
#
# Camera discipline is identical for all three: stock Camera, landscape, 1x,
# fill the frame with the screen, hit record, tap-hold to lock AE/AF during
# the countdown, do NOT touch the exposure slider, then hold still.
cd "$(dirname "$0")"
MODE="$1"

dock() {
  osascript -e "tell application \"System Events\" to tell dock preferences to set autohide to $1" \
            -e "tell application \"System Events\" to tell dock preferences to set autohide menu bar to $1" 2>/dev/null
}

quiet_start() {
  # A notification banner landing on the panel mid-take is opaque pixels on
  # payload cells, and it steals focus from the transmitter. The background
  # decodes/renders THIS project spawns are the usual source, so stop them for
  # the duration rather than hoping they stay silent.
  pkill -f "poc/fast_decode.py"     2>/dev/null
  pkill -f "poc/analyze_"           2>/dev/null
  pkill -f "poc/make_"              2>/dev/null
  cat <<'QEOF'
  >> Turn on Focus / Do Not Disturb now (Control Centre > Focus).
  >> macOS 15 has no scriptable switch for it without a user shortcut, so
  >> this is the one manual step. A banner over the code voids the take.
QEOF
}

case "$MODE" in
density)
  cat <<'EOF'
------------------------------------------------------------------
  DENSITY LADDER.  Camera: 4K at 60 fps.
  Four cell sizes in one take: 12/11/10/9 px, 7s each per loop.
  12px is the control; if it reads badly the take is void.
  22s countdown, then 3 loops x 28s (~106s).
------------------------------------------------------------------
EOF
  quiet_start; sleep 6; dock true
  ffplay -v error -fs -alwaysontop -noborder -autoexit demo/_tx_density_lead.mp4
  ffplay -v error -fs -alwaysontop -noborder -loop 3 -autoexit demo/_tx_density.mp4
  dock false
  echo "AirDrop it, then:  python3 poc/analyze_density.py ~/Downloads/IMG_XXXX.MOV"
  ;;
strobe)
  cat <<'EOF'
------------------------------------------------------------------
  STROBE.  Camera: 4K at 60 fps.  SCREEN BRIGHTNESS: MAXIMUM.
  Code on even refreshes, black on odd. Half the light reaches the
  sensor, which is why brightness must be maxed: the AE lock needs
  to land at a SHORT exposure (<= 1/120 s) or flashes mix again.
  Rolling shutter may band the frame; dark rows are erasures, not
  errors, and the fountain absorbs them. The capture self-reports
  whether the exposure condition held.

  Safari opens at the transmitter. Click for fullscreen; the take
  starts on the fullscreen transition. Watch the HUD: it must say
  ~120fps. Record through the countdown + ~5 loops, stop when DONE.
------------------------------------------------------------------
EOF
  quiet_start; sleep 6
  pkill -f "http.server 8000" 2>/dev/null
  (python3 -m http.server 8000 >/tmp/heliogram_httpd.log 2>&1 &)
  sleep 1; dock true
  open -a Safari "http://localhost:8000/tx120.html?strobe=1&loops=5"
  echo "server up; press enter here when the take is done to stop it"
  read -r; pkill -f "http.server 8000"; dock false
  echo "AirDrop it, then:  python3 poc/fast_decode.py ~/Downloads/IMG_XXXX.MOV /tmp/out.bin --grid 252x163 --ecc 48 --subblock --soft --scan --full"
  ;;
120)
  cat <<'EOF'
------------------------------------------------------------------
  120 FPS.  Camera: 4K at ONE HUNDRED TWENTY fps. This is the one
  setting that decides the take: Settings > Camera > Record Video
  > 4K/120. At 60 the display outruns the camera and nothing will
  decode. Screen brightness HIGH (half the exposure per frame).

  Safari opens at the transmitter. Click for fullscreen; watch the
  HUD: it must say ~120fps. Record countdown + ~6 loops.
------------------------------------------------------------------
EOF
  quiet_start; sleep 6
  pkill -f "http.server 8000" 2>/dev/null
  (python3 -m http.server 8000 >/tmp/heliogram_httpd.log 2>&1 &)
  sleep 1; dock true
  open -a Safari "http://localhost:8000/tx120.html?loops=6"
  echo "server up; press enter here when the take is done to stop it"
  read -r; pkill -f "http.server 8000"; dock false
  echo "AirDrop it, then:  python3 poc/fast_decode.py ~/Downloads/IMG_XXXX.MOV /tmp/out.bin --grid 252x163 --ecc 48 --subblock --soft --scan --full"
  ;;
*)
  echo "usage: ./film.sh density | strobe | 120"; exit 1 ;;
esac
