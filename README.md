# screen-camera

Screen-to-camera file transfer, aimed past the current open-source ceiling
(decimen-optical-transfer, 129.2 KB/s) toward the physics ceiling of a
120Hz display filmed by a modern iPhone. Research brief and prior art live
in the Obsidian vault: `wiki/projects/screen-camera/`.

## Layout

```
experiments/smear-test/   THE deciding experiment. Run this first.
  SmearTest.swift         120Hz Metal test patterns (build.sh -> ./smeartest)
  analyze_smear.py        verdict on iPhone 240fps captures: CLEAN/MARGINAL/SMEARED
  PROTOCOL.md             the 20-minute procedure

poc/                      Phase-0 measurement rig, working end-to-end
  encode.py               file -> animated grid-code mp4 (mono or 8-color)
  simulate.py             synthetic camera channel (perspective/blur/noise/straddle)
  decode.py               video -> file + channel stats
  codec/fountain.py       LT fountain, peeling + GF(2) elimination fallback
  codec/grid.py           frame format: finders, RS-protected header + payload
```

## Verified results (software loopback through the simulated camera)

| Config | Goodput | Notes |
|---|---|---|
| mono 120x68 @ 30fps | 14.1 KB/s | baseline sanity |
| color8 120x68 @ 30fps | 59.8 KB/s | |
| color8 180x100 @ 30fps | 108.5 KB/s | ~5.9 camera px/cell |
| mono, 50% frame straddle | goodput halves | the utilization problem, quantified |

All runs recover the file bit-exact (sha256-verified). The simulator is
kinder than a real camera: no moiré, no rolling-shutter skew, no phone
H.264/HEVC artifacts, no focus hunting. Real capture numbers come after
the smear test.

Scaling that is real but unproven until filmed: 108.5 KB/s x (120/30) fps
≈ 434 KB/s ≈ 3.6 Mbps, before straddle-aware decoding and denser grids.

## Quickstart

```
# end-to-end software loopback
python3 poc/encode.py FILE clean.mp4 --mode color8
python3 poc/simulate.py clean.mp4 dirty.mp4
python3 poc/decode.py dirty.mp4 recovered.bin

# real transmission: loop clean.mp4 fullscreen, film with iPhone slo-mo,
# AirDrop the capture, then decode.py the capture directly.
```

## Design notes

- **Record-then-decode.** Every existing tool decodes live and pays a
  real-time CPU tax in the loop (cimbar's effective fps is decode-bound).
  We film with the iPhone at 240fps, then decode offline. Nobody does this.
- **Per-block CRC32 inside the RS payload.** RS(255,223) mis-corrects
  heavily damaged codewords into valid-but-wrong ones; one poisoned block
  silently corrupts the entire fountain output. Found by the straddle
  stress test, fixed with 4 bytes/frame.
- **`cell_margin` in decode stats** is the soft-decision hook: when the
  margin distribution sits below ~0.15, hard thresholding throws away
  exactly the information a soft LT decoder could still use.
