# The one-evening experiment

This decides the project. Total time: ~20 minutes.

Warning: the test patterns flash the full screen at 60Hz. If anyone
photosensitive is in the room, don't run it.

## Steps

1. Build and run:

   ```
   ./build.sh && ./smeartest
   ```

   The HUD (top-left) must read ~120 fps in green. If it reads 60, check
   System Settings → Displays → Refresh Rate is "ProMotion", and quit
   low-power mode. Do not proceed at 60 — the whole question is 120.

2. Dim the room. Screen bright. The phone must be forced to a short
   exposure; ambient light is the enemy of this measurement.

3. On the iPhone: Camera → Slo-mo → 240 fps (Settings → Camera → Record
   Slo-mo → 1080p at 240 fps). Prop the phone ~40-60cm away, screen
   filling most of the frame, square-on. Lock focus/exposure by
   long-pressing on the screen image (AE/AF LOCK must show).

4. Record ~10 seconds of each:
   - pattern 1 (`alternate`) — the main measurement
   - pattern 2 (`split`)     — local-dimming coupling check
   - pattern 6 (`static`)    — reference

5. AirDrop the three videos to the Mac, then:

   ```
   python3 analyze_smear.py alternate.mov
   python3 analyze_smear.py split.mov --pattern split
   ```

## Reading the result

- **CLEAN** — captured frames show sharp black/white horizontal bands.
  The channel carries sub-frame structure. Phase 3 (rolling-shutter-aware
  decode) is viable on this panel.
- **MARGINAL** — bands visible but boundaries eat 12-25% of the frame.
  Usable with soft-decision decoding; expect reduced gains.
- **SMEARED** — boundaries are gray mush. The mini-LED panel (or the
  camera exposure) low-pass-filters the alternation. Next step: re-run
  with the roles swapped — smeartest pattern on an iPhone (OLED, ~instant
  response) filmed by whatever second camera is available, or accept
  60Hz symbol rate on the Mac panel and put the speed into density
  instead.

One subtlety: the analyzer measures the *channel* (panel response +
backlight + camera exposure + rolling shutter together). That is the
correct thing to measure — the decoder will face exactly this channel,
and it doesn't matter which component is at fault if the boundary is mush.
The `split` pattern separates one factor: if the static gray half ripples
while the top half flashes (`static_region_ripple` > ~0.02), the
local-dimming backlight is coupling zones together and even 60Hz static
codes will carry zone-level noise.
