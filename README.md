# heliogram

Send a file from your laptop to your phone as light, no network involved.
**208.2 KB/s, hand-held, bit-exact**, verified against decimen-optical-transfer
(128 KB/s hand-held, 186 KB/s propped).

A heliograph is the instrument that sent messages across air gaps by flashing
sunlight off a mirror. A heliogram is the message. This is the same idea with a
retina display and a CMOS sensor: the file is rendered as an animated 2-D
barcode, played fullscreen, filmed with a phone, and decoded back from the
video.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (`brew install ffmpeg` / `apt
  install ffmpeg`)
- A laptop with a screen and a phone with a camera

## Install

```bash
git clone https://github.com/ofitzharding/heliogram.git
cd heliogram
python3 -m pip install -r requirements.txt
```

## Usage

**1. Encode and display the file.** Run this on the machine that has the file:

```bash
python3 send.py path/to/your/file
```

This renders the file into an animated grid code and plays it fullscreen: a
~22 second lock-in countdown, then five loops of the code (~10s per loop for a
small file, longer for a bigger one). Pass `--no-play` to just build the video
without opening it, if you want to film it later or transfer it to another
machine.

**2. Film it.** On your phone: stock Camera app, 4K at 60fps, landscape, 1x
zoom, fill the frame with the screen. Hit record, then during the countdown
tap-and-hold on the screen to lock autoexposure/autofocus. **Don't touch the
exposure slider** — it cuts exposure without raising ISO, which underexposes
the shot. Then hold the phone still until the code stops playing.

**3. Get the video back onto your laptop** (AirDrop, cable, whatever) and
decode it:

```bash
python3 recv.py ~/Downloads/IMG_XXXX.MOV
```

This writes the file under its original name in the current directory (use
`--out DIR` to choose where) and prints whether the recovered bytes' sha256
matched the original:

```
file      your_file.pdf
size      1,121,502 bytes
sha256    VERIFIED - byte-identical to the original
written   ./your_file.pdf
```

If the take was rough, `poc/quickcheck.py` gives a pass/fail verdict on a
capture in about a minute instead of running the full decode (which can take
several minutes on a long take):

```bash
python3 poc/quickcheck.py ~/Downloads/IMG_XXXX.MOV --grid 252x163
```

### Try it without filming anything

`demo/kitten.png` is a ready-made test payload. Encode it, decode the video
you just rendered directly (no camera involved), and confirm the round trip:

```bash
python3 send.py demo/kitten.png --no-play
python3 recv.py demo/_send.mp4 --out /tmp
diff demo/kitten.png /tmp/kitten.png && echo "byte-identical"
```

## How it works

3024x1964 laptop panel at 60 Hz, 252x163 mono cells at 12 px/cell (99.6% of the
panel). Each frame carries 19 RS(255,207) codewords; each codeword is its own
CRC32-protected fountain symbol, so a frame damaged in one band still
contributes the others. Channel ceiling 226.0 KB/s. The file's name and sha256
travel inside the payload (`poc/container.py`), so the receiver can write it
back under its own name and verify it without any side channel.

Measured end to end on real hand-held footage, every figure sha256-verified
against the source:

| capture | full span | best window |
|---|---|---|
| IMG_7924, 378x245 strobe rung, hand-held | **229.7 KB/s** | 229.7 KB/s |
| IMG_7920, 336x218 strobe rung, hand-held | 142.5 KB/s | 142.5 KB/s |
| IMG_7872, 252x163, hand-held | 131.0 KB/s | **208.2 KB/s** |
| earlier baseline take, 252x140 | 153.2 KB/s | 180.4 KB/s |
| clean transmit file (no camera in the path) | 224.3 KB/s | — |

*Full span* is file size over the time from the first frame that contributed a
symbol to the frame that completed the file. *Best window* is the shortest
contiguous window from which the whole file decodes, verified by decoding from
that window alone and comparing bytes; it is selected after the fact, so both
numbers are quoted.

### Receiver pipeline

1. Finder detection, best 4-subset by parallelogram/aspect score, homography.
2. Radial coefficient, hill-climbed per frame on certified-codeword count.
3. 3x3 box mean at each cell centre.
4. **Local decision threshold**: occupancy-normalised box mean over a 15x15
   *cell* neighbourhood. RS+fountain output is pseudorandom, so its local mean
   converges on the midpoint of the eye at that point on the screen. Replacing
   one global Otsu with this doubled codeword yield (19.8% -> 39.2%).
5. Per-codeword RS, then soft-erasure RS with erasures placed by confidence,
   then the CRC32 gate. Erasures are the bootstrap, not an optimisation:
   without them no frame ever certifies enough to become a kernel donor and the
   equalizer contributes nothing at all.
6. **CAG** (below) on codewords still missing.
7. Tile-PRML with a rolling kernel donor refitted on certified cells only,
   skipped on frames that already certify in full.
8. Fountain assembly; stop as soon as the file closes.

## The interesting part

**The a-priori pilots are a biased estimator of sampling geometry.** Sweeping
the radial coefficient over 16 frames and asking two questions of each:

| chosen by | median k1 | codewords recovered |
|---|---|---|
| pilots (finder patterns, timing ring, separators) | +0.0075 | 19 |
| the code (certified codewords) | +0.0238 | 150 |

0 of 16 frames agree, the offset is one-signed on every frame, and trusting the
pilots costs **87.3% of codewords**. That is bias, not noise, and the cause is
structural: finders, ring and separators all lie on the grid *border*, so they
sit at a different radius from the payload while the parameter being estimated
is radius-dependent. Screen-camera decoders generally estimate geometry exactly
this way — corner detection, Hough lines, neural segmentation.

**Codeword-adjudicated geometry (CAG).** A codeword carries CRC32 behind
Reed-Solomon, which is a free, exact, zero-false-accept test for "did I sample
this band on grid?". So geometry stops being a parameter estimated once per
frame from the wrong evidence and becomes a *search*, run per codeword, against
the only unbiased spatially-local evidence available: the code itself. A
candidate either yields a certified codeword — correct by construction,
whatever produced it — or is discarded. The search cannot do harm, so it is
bounded only by compute, and needs no ground truth. Measured in isolation:
42.2% -> 51.3% of codewords, 1.22x.

**Prior art, stated honestly.** Using the channel decoder to refine
synchronisation is an established field — code-aided and turbo synchronisation
go back to the late 1990s, typically SISO soft output driving a continuous
estimate of a global 1-D timing or phase parameter. CAG differs in form (a hard
zero-false-accept oracle, a discrete search, 2-D spatial geometry, per-codeword
granularity) but it is an instantiation of that family, not a new principle.
What has not been found stated elsewhere is the *measurement*: that structural
pilots are biased in this regime, and by how much. A deeper prior-art pass is
outstanding.

## What did not work, and why

Kept because the negative results cost more to find than the positive ones.

- **Interleaving is 7.7x worse, not better.** Per-codeword errors ramp down the
  frame, so spreading each codeword over the whole frame to equalise them is
  the obvious move. Yield goes 38.1% -> 5.0%. RS failure is a *threshold* and
  the mean error count already exceeds the budget, so Jensen runs the other
  way: you want errors *concentrated*, piled into sacrificial codewords so the
  rest come in under budget. Interleaving is reflexive in communications and it
  is wrong on any channel operating above its own correction threshold.
- **Denser grids are dead, for a stated reason.** 350x194 *does* sample on grid
  (97% structure-cell agreement at k1=+0.0025; the earlier "0 codewords" had
  been measured with the 252 grid's geometry) and still certifies 0/32 on 124
  frames, because 9.0 camera-px/cell is ~3% raw BER against a 24-byte budget.
- **No rolling-shutter tear.** Mid-eye cell fraction ramps monotonically 9% ->
  23% top to bottom, identically on every frame: a stable spatial residual, not
  a temporal step.
- **Reusing the previous frame's homography, gated on the header, loses yield.**
  10% fewer blocks for 5% less wall time — the header tolerates far more
  geometric drift than the payload does. Off by default.

## Limits

- Decoding runs at ~3 fps on 4K, so the file is reconstructed on the laptop
  after the video comes back, not on the phone. Reed-Solomon alone is 15.5% of
  a frame in pure Python. Decoder throughput does not affect any KB/s figure
  here — goodput is measured on the capture's clock, not the decoder's.
- One device pair, one panel, one payload size.
- Best-window figures are selected after the fact; full-span is quoted beside
  them.

## Project layout

```
send.py  recv.py        the demo
poc/fast_decode.py      parallel receiver
poc/softdec.py          certified-label receiver, one frame at a time
poc/codec/grid.py       layout, render, locate, sample, demodulate
poc/codec/fountain.py   LT fountain, peeling + GF(2) elimination fallback
poc/make_record.py      transmit builder
poc/quickcheck.py       judge a take in a minute instead of twenty
poc/exp_*.py            the experiments behind every claim above
demo/CREDITS.md         payload image licences
```

## License

[MIT](LICENSE).
