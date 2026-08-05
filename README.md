# heliogram

Send a file from one screen to another device's camera. No network, no
pairing, no account, and no install on either side.

**Live demo: [ofitzharding.github.io/heliogram](https://ofitzharding.github.io/heliogram/)**

A heliograph is the instrument that sent messages across air gaps by flashing
sunlight off a mirror. A heliogram is the message. This is the same idea with a
retina display and a CMOS sensor: the file is rendered as an animated 2-D
barcode, played fullscreen, filmed, and decoded back from the pixels. The
file's name and sha256 travel inside the payload, so the receiver names the
file itself and can prove the bytes are right without a back-channel.

Fastest verified transfer to date: **229.7 KB/s, hand-held, bit-exact**, from a
60 Hz laptop panel to a stock phone camera shooting plain 4K60.

## Two ways to run it

| | path | install | status |
|---|---|---|---|
| **Browser pair** | `hgtx.html` on the screen, `rx.html` on the phone, live camera, decode in the page | none, each end is one HTML file | transfer verified bit-exact; throughput on the current grid not yet measured |
| **Film pipeline** | `send.py` on the screen, phone films it, `recv.py` decodes the video | Python + ffmpeg on the receiving machine | every number in the results table below |

They are different machines. The browser pair trades most of the pixels for
zero install; the film pipeline spends a full 4K capture and a heavier decoder
to get the rate up.

## Quick start: browser pair

Nothing to install. Both pages are static.

1. Open [the demo](https://ofitzharding.github.io/heliogram/) on a laptop or
   monitor, press **SEND**, and drop in a file.
2. Press **open the receiver QR**, point the phone's camera at the code, and
   open the link. (Camera access needs HTTPS, which the demo URL provides. To
   serve the pages yourself over a LAN, put them behind TLS:
   `tailscale serve --bg --https=443 localhost:8000` works.)
3. Press `F` on the sender for fullscreen. During the white countdown, aim the
   phone at the screen and tap to lock focus and exposure.
4. Fill the phone's frame with the code and hold still. The file rebuilds live
   and checks its own sha256.

To run the pages locally:

```bash
python3 -m http.server 8000
```

The receiver carries its own proof: **RS SELF-TEST** checks its Reed-Solomon
implementation against vectors from the python decoder (40/40 byte-identical),
**OPTICS SELF-TEST** runs a synthetic camera view of a real transmit frame
through the actual detection path, and the loopback test drives sender-encoded
frames through the receiver until the fountain closes. All three run in the
page with no camera involved.

## Quick start: film pipeline

Requirements: Python 3.9+, [ffmpeg](https://ffmpeg.org/) on your `PATH`, a
screen, and a phone camera.

```bash
git clone https://github.com/ofitzharding/heliogram.git
cd heliogram
python3 -m pip install -r requirements.txt
```

**1. Display the file.** On the machine holding it:

```bash
python3 send.py path/to/your/file
```

This renders the file into the grid code and plays it fullscreen: a ~22 second
lock-in countdown, then five loops of the code (~10s per loop for a small
file). `--no-play` builds the video without opening it.

**2. Film it.** Stock camera app, 4K at 60fps, landscape, 1x zoom, fill the
frame with the screen. Hit record, then tap and hold on the screen during the
countdown to lock autoexposure and autofocus. Do not touch the exposure slider:
it cuts exposure without raising ISO, which underexposes the shot. Hold still
until the code stops playing.

**3. Decode.** Get the video onto the laptop (AirDrop, cable, whatever):

```bash
python3 recv.py ~/Downloads/your-capture.MOV
```

The file lands under its original name in the current directory (`--out DIR` to
choose where), with a verdict on the bytes:

```
file      your_file.pdf
size      1,121,502 bytes
sha256    VERIFIED - byte-identical to the original
written   ./your_file.pdf
```

For a rough take, `src/quickcheck.py` returns a pass/fail on a capture in about
a minute instead of running the full decode:

```bash
python3 src/quickcheck.py ~/Downloads/your-capture.MOV --grid 252x163
```

### Try it without filming anything

`demo/kitten.png` is a ready-made payload. Encode it, decode the rendered video
directly with no camera in the path, and compare:

```bash
python3 send.py demo/kitten.png --no-play
python3 recv.py demo/_send.mp4 --grid 252x163 --out /tmp
diff demo/kitten.png /tmp/kitten.png && echo "byte-identical"
```

`recv.py` walks the density ladder from the densest grid down when it is not
told which one to expect, so `--grid` saves it several minutes whenever you
already know.

## Results

Every figure below is a wall-clock rate over a transfer whose sha256 matched
the source. *Full span* is file size over the time from the first frame that
contributed a symbol to the frame that completed the file. *Best window* is the
shortest contiguous window from which the whole file decodes, verified by
decoding from that window alone; it is selected after the fact, so both numbers
are quoted together.

| configuration | full span | best window |
|---|---|---|
| **378x245 at 8 px/cell, strobed, 4K60 hand-held** | **229.7 KB/s** | |
| 252x163 at 12 px/cell, 4K60 hand-held (the `send.py` profile) | 131.0 KB/s | 208.2 KB/s |
| 252x140, earlier baseline take | 153.2 KB/s | 180.4 KB/s |
| clean transmit file, no camera in the path | 224.3 KB/s | |

The record take carried a 1.1 MB payload from the browser transmitter with
black-frame strobing (the display becomes the shutter, so an exposure that
straddles a boundary integrates code plus black instead of two codes), decoded
with the pool-level ghost sweep in `src/fast_decode.py`. The 252x163 profile is
what `send.py` ships, because it needs neither a strobe nor a browser.

The browser pair's first end-to-end transfer moved a 16,695 byte `.docx` from a
laptop to an iPhone through Safari, sha256 verified, at 8.4 KB/s. That run used
a 160x104 grid with corner detection re-running on every frame. The shipped
grid is now 320x208 (30 to 31 codewords per frame, a ceiling near 180 KB/s at
4K30) with geometry tracking between frames, measured at 18 ms/frame in the
page. Neither change has faced a camera yet, so no rate is claimed for them.

## How it works

3024x1964 laptop panel at 60 Hz, 252x163 mono cells at 12 px/cell (99.6% of the
panel). Each frame carries 19 RS(255,207) codewords, and each codeword is its
own CRC32-protected fountain symbol, so a frame damaged in one band still
contributes the others. Channel ceiling 226.0 KB/s.

### Receiver pipeline

1. Finder detection, best 4-subset by parallelogram/aspect score, homography.
2. Radial coefficient, hill-climbed per frame on certified-codeword count.
3. 3x3 box mean at each cell centre.
4. **Local decision threshold**: occupancy-normalised box mean over a 15x15
   *cell* neighbourhood. RS+fountain output is pseudorandom, so its local mean
   converges on the midpoint of the eye at that point on the screen. Replacing
   one global Otsu with this doubled codeword yield (19.8% to 39.2%).
5. Per-codeword RS, then soft-erasure RS with erasures placed by confidence,
   then the CRC32 gate. Erasures are the bootstrap, not an optimisation:
   without them no frame ever certifies enough to become a kernel donor and the
   equalizer contributes nothing at all.
6. **CAG** (below) on codewords still missing.
7. Tile-PRML with a rolling kernel donor refitted on certified cells only,
   skipped on frames that already certify in full.
8. Fountain assembly, stopping as soon as the file closes.

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
this way, through corner detection, Hough lines, or neural segmentation.

**Codeword-adjudicated geometry (CAG).** A codeword carries CRC32 behind
Reed-Solomon, which is a free, exact, zero-false-accept test for "did I sample
this band on grid?". Geometry stops being a parameter estimated once per frame
from the wrong evidence and becomes a *search*, run per codeword, against the
only unbiased spatially-local evidence available: the code itself. A candidate
either yields a certified codeword, correct by construction whatever produced
it, or is discarded. The search cannot do harm, so it is bounded only by
compute and needs no ground truth. Measured in isolation: 42.2% to 51.3% of
codewords, 1.22x.

**Content-identity ghost detection.** Rolling shutter makes a capture frame
straddle two code frames, and the later index then carries the earlier frame's
rows. Because a fountain symbol's content is unique to about 2^-1600, an index
whose bytes equal the same codeword position one sequence earlier is a straddle
ghost by identity: no thresholds, no optical flow, nothing the receiver is not
already holding. Sweeping those ghosts out of the symbol pool took the 8 px
rung from corrupt to bit-exact, and then from 170.4 KB/s to 229.7 KB/s, because
ghost indices had been occupying symbol slots the fountain then had to out-wait.

**Prior art, stated honestly.** Using the channel decoder to refine
synchronisation is an established field: code-aided and turbo synchronisation
go back to the late 1990s, typically with SISO soft output driving a continuous
estimate of a global 1-D timing or phase parameter. CAG differs in form (a hard
zero-false-accept oracle, a discrete search, 2-D spatial geometry, per-codeword
granularity) but it is an instantiation of that family, not a new principle. An
independent literature sweep (Gemini deep research, August 2026) placed CAG
alongside Zagaynov US 11,960,966 and Digimarc US 6,792,542, and found no prior
statement of the *measurement* here: that structural pilots are biased in this
regime, and by how much. The same sweep found no prior statement of CRC-gated
equalizer training, of content-identity ghost detection, or of the
dual-sequence rolling harvest applied to dense 2-D matrix codes. That is a
literature search, not a patent opinion.

## What did not work, and why

Kept because the negative results cost more to find than the positive ones.

- **Interleaving is 7.7x worse.** Per-codeword errors ramp down the frame, so
  spreading each codeword over the whole frame to equalise them is the obvious
  move. Yield goes 38.1% to 5.0%. RS failure is a *threshold* and the mean
  error count already exceeds the budget, so Jensen runs the other way: you
  want errors *concentrated*, piled into sacrificial codewords so the rest come
  in under budget. Interleaving is reflexive in communications and it is wrong
  on any channel operating above its own correction threshold.
- **Denser grids have a stated failure mode.** 350x194 *does* sample on grid
  (97% structure-cell agreement at k1=+0.0025; the earlier "0 codewords" had
  been measured with the 252 grid's geometry) and still certifies 0/32 on 124
  frames, because 9.0 camera-px/cell is ~3% raw BER against a 24-byte budget.
  Density pays only when the framing pays for it, which is why the record rung
  runs 8 px cells at full-sensor framing.
- **No rolling-shutter tear.** Mid-eye cell fraction ramps monotonically 9% to
  23% top to bottom, identically on every frame: a stable spatial residual, not
  a temporal step.
- **Colour loses at this cell size.** 8-colour constellations measure 1.0 sigma
  separation, and colour4 needs ~20 px cells where mono runs at 10 to 12. Dense
  mono beats sparse colour.
- **Reusing the previous frame's homography, gated on the header, loses yield.**
  10% fewer blocks for 5% less wall time. The header tolerates far more
  geometric drift than the payload does. Off by default.

## Limits

- Decoding runs at ~4 fps per worker on 4K, so the film pipeline reconstructs
  the file on the laptop after the video comes back, not on the phone. Decoder
  throughput does not affect any KB/s figure here: goodput is measured on the
  capture's clock, not the decoder's.
- One device pair, one panel, a handful of payload sizes.
- Best-window figures are selected after the fact, and full-span is quoted
  beside them.
- AirDrop over a good link is tens of MB/s and this will never approach it. The
  cases this serves are the ones AirDrop cannot: air-gapped machines,
  cross-ecosystem transfers with no account and no pairing, and one-directional
  transfer where the sending side must receive nothing at all.

## Related work

[decimen-optical-transfer](https://github.com/bashalarmistalt/decimen-optical-transfer)
showed that screen-to-camera transfer belongs in a browser tab, sending QR v40
frames at 2953 bytes per frame from a 120 fps ProMotion display, and it
published the hand-held figures that gave this work something concrete to
measure against: 128 KB/s hand-held, 186 KB/s propped. Worth reading if this
problem interests you. Every figure in this README is heliogram's own, measured
on a 60 Hz panel under the conditions stated beside it.

## Repository layout

```
index.html            landing page for the hosted demo
hgtx.html             browser sender: drag a file in, it becomes light
rx.html               browser receiver: camera in, file out, sha256 checked
qr.html               dependency-free QR of the receiver URL

send.py  recv.py      the film pipeline, end to end

src/codec/grid.py     layout, render, locate, sample, demodulate
src/codec/fountain.py LT fountain, peeling + GF(2) elimination fallback
src/container.py      name/size/sha256 wrapper carried inside the payload
src/fast_decode.py    parallel receiver
src/softdec.py        certified-label receiver, one frame at a time
src/make_record.py    transmit builder behind send.py
src/make_web_rx.py    builds the web pages' metadata and transmit dirs
src/quickcheck.py     judge a take in a minute instead of twenty
src/crs2.c            C Reed-Solomon inner loop
src/exp_*.py          the experiments behind every claim above

scripts/              filming runbooks and session drivers
lab/                  research transmitters (grid ladders, strobe, playlists)
docs/                 research brief and background
demo/                 test payloads and generated transmit dirs
experiments/          standalone instruments (rolling-shutter smear test)
```

## License

[MIT](LICENSE).
