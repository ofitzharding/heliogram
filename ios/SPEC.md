# heliogram iOS receiver — build specification

This document is the complete contract for building the native iOS receiver.
It is written to be executed by an agent without access to this project's
history. Follow it in order. Do not improvise on wire formats, constants, or
test gates: every one of them is measured or extracted from a working system,
and the browser sender you must interoperate with WILL NOT CHANGE.

## 0. Mission and definition of done

Build an iOS app that receives a file transmitted by the EXISTING browser
sender (`hgtx.html`, served from this repo, unmodified) by pointing the
iPhone's camera at the laptop screen.

DONE means, in this order, with no step skipped:

1. Gates G1-G5 (section 8) pass as XCTests **in the iOS simulator**.
2. A live hand-held transfer of a file >= 1 MB completes with the app
   printing `BIT-EXACT` and a sha256 that matches the source file.
3. Full-span rate >= 100 KB/s, where full-span = file bytes divided by
   (time of fountain close minus time of first certified symbol).

Report rates ONLY as measured by criterion 3. Never report a rate for a
transfer that did not verify sha256. This is a project law.

## 1. Non-goals

- No sending from iOS in v1 (the laptop browser page is the sender).
- No App Store polish, no settings screens, no onboarding.
- Do not modify anything outside the `ios/` directory except this repo's
  `ios/FIELDLOG.md` (which you create and append results to).
- Do not touch `hgtx.html`, `rx.html`, `qr.html`, or anything in `src/`.

## 2. What already exists and where

| artifact | path | role |
|---|---|---|
| C decode core | `src/crs2.c` | GF(256) RS + CRC32 + certify. Compiles into the app UNCHANGED. |
| layout metadata | `demo/webtxweb/rxmeta.slim.json` | grid geometry, cell orders, header masks, RS test vectors. Bundle it as an app resource. |
| fountain fixtures | `ios/fixtures/fountain_vectors.json` | ground truth for the LT fountain PRNG, generated under JavaScriptCore/Darwin libm (bit-exact with a Safari sender). |
| header fixtures | `ios/fixtures/header_vectors.json` | exact post-RS post-mask header bytes for 4 (seq,k,blockSize,fileSize) tuples. |
| reference receiver | `rx.html` | the browser receiver; the algorithms you are porting, in shipped, field-tested form. When this spec and rx.html disagree on an algorithm detail, rx.html wins. |

## 3. The wire format (interop contract — memorize, never modify)

The sender renders one "code frame" (`seq` = frame number) as a grid of
`gw x gh = 320 x 208` black/white cells. All multi-byte integers are
LITTLE-ENDIAN. All bit-unpacking of bytes onto cells is MSB-FIRST.

### 3.1 rxmeta.slim.json fields you consume

- `gw`, `gh`: grid size (320, 208).
- `n_sub`: codewords per frame (31).
- `sub`: payload bytes per codeword (203).
- `ecc`: RS parity bytes per codeword (48). Codeword length is 255.
- `hdr_len_pre` = 28, `hdr_ecc` = 40, `hdr_len_enc` = 68, `hdr_phases` = 8.
- `finders`: 4 grid-space points `[tl, tr, bl, br]`, the finder centres
  (e.g. x = 4.5 cells from the edge). Order matters: homographies map
  `finders[i] -> image quad[i]`.
- `struct_cells` (int array) + `struct_vals` (base64): cell indices and a
  bit-packed value stream for the static structure (finders, timing ring,
  separators). You never decode these; they exist for rendering synthetic
  frames in tests.
- `header_cells` (int array): cell index for header bit i*8+b of byte i.
  Only the first `hdr_len_enc*8` entries are written by the sender.
- `payload_cells` (int array): cell index for payload bit
  `(j*255 + i)*8 + b` of byte i of codeword j.
- `hdr_masks`: base64 array of `hdr_phases` XOR masks, each
  `hdr_len_enc` bytes.
- `rs_vectors`: 40 RS test vectors (see gate G1).

Cell index = `row * gw + col`.

### 3.2 Codeword

```
msg[0..3]   = crc32(block) little-endian     (crc32 = zlib polynomial,
                                              reflected 0xEDB88320,
                                              init & final xor 0xFFFFFFFF)
msg[4..206] = block (sub = 203 bytes, one fountain symbol)
codeword    = RS_encode(msg, nsym=48)        -> 255 bytes, systematic,
                                              GF(256) prim poly 0x11d,
                                              first consecutive root fcr=0,
                                              generator element 2
```
Certification = RS decode succeeds AND crc32 matches. This is a
zero-false-accept oracle; the entire receiver philosophy rests on it:
**no geometry, no sequence guess, no anything is trusted unless a
certification says so.**

### 3.3 Header (28-byte body, RS(68, 28), 8-phase whitening)

Byte layout of the body (offsets):

```
0..3   magic "SCPC" (0x53 0x43 0x50 0x43)
4      version = 2
5      mode = 0 (MODE_MONO)
6..9   seq        u32le
10..13 k          u32le   (fountain source-block count)
14..15 blockSize  u16le   (= sub = 203)
16..19 fileSize   u32le   (wrapped container length in bytes)
20     ecc (= 48)
21, 22 zero
26..27 crc16      u16le   = crc32(body[0..25]) & 0xFFFF
```
Then `enc = RS_encode(body, nsym=hdr_ecc=40)` (68 bytes), then XOR with
`hdr_masks[seq % 8]`. Receiver tries all 8 masks; a decode is accepted only
if magic, crc16, AND `seq % 8 == mask index` all hold.
`ios/fixtures/header_vectors.json` has 4 exact encoded outputs; your packer
and parser must round-trip them byte-identically.

### 3.4 Fountain (LT, robust soliton)

The file travels as a container (3.5), zero-padded to `K * sub` where
`K = ceil(len/sub)`. Fountain symbol index `idx = seq * n_sub + j` for
codeword j of frame seq. Symbol content = XOR of source blocks selected by
`blockIndices(idx, K)`:

```js
function mulberry32(a){ return function(){
  a |= 0; a = (a + 0x6D2B79F5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
// robust soliton c=0.03 delta=0.5; cum = cumulative pmf array (1-indexed)
// blockIndices: rng = mulberry32(idx ^ 0x9E3779B9); u = rng();
// degree d = first index with cum[d] >= u (linear scan from 1);
// then draw indices i = floor(rng()*k) % k, skipping repeats, until d unique.
```
Port to Swift with `Double`, `Foundation.log`, `Foundation.sqrt`, and 32-bit
wrapping integer ops replicating `Math.imul` (`&*` on Int32/UInt32). On
Darwin, JavaScriptCore and Swift share libm, so results are bit-exact with a
Safari sender; `ios/fixtures/fountain_vectors.json` is the proof and gate G3
enforces it, including `Double.bitPattern` equality on the cum probes.

Decoder: belief-propagation peeling exactly as `rx.html` (`fountainAdd`,
`resolve`): XOR out already-decoded sources; degree-1 resolves and cascades
through a pending list. Peeling alone closes k=5166 with ~4% overhead
(measured); no Gaussian elimination needed in v1.

### 3.5 Container ("HGC1")

```
0..3          "HGC1"
4..5          nameLen u16le
6..6+n-1      name (utf8)
6+n..9+n      size u32le (payload bytes)
10+n..41+n    sha256 of payload (32 raw bytes)
42+n..        payload
```
After the fountain closes: take first `fileSize` bytes of the concatenated
source blocks, unwrap, hash the payload, compare with the embedded sha256.
That comparison is the only thing allowed to print BIT-EXACT.

### 3.6 Sender pacing (what the camera will see)

The sender paces code frames on the wall clock (`?rate=`). LAW, measured
twice tonight in the field: **the seq rate must never integer-divide or
equal the capture frame rate**, or the flip phase parks inside every
exposure and each capture is an undecodable two-frame blend for tens of
seconds (the clocks drift at ppm rates). For the app at 4K60 capture, drive
the sender at `hgtx.html?rate=56`. Ceiling at 56 seq/s:
`31 * 203 * 56 = 352.4 KB/s`. Your 100 KB/s target is 28% of ceiling;
the film pipeline achieves 58% efficiency offline, the browser receiver
~10-20% live. Native should land between.

## 4. Repository layout to create (everything under `ios/`)

```
ios/
  project.yml                  XcodeGen manifest (below)
  Heliogram/
    App.swift                  @main SwiftUI App
    ReceiverView.swift         preview + HUD + verdict UI
    CaptureController.swift    AVCaptureSession config + frame delivery
    DecodeEngine.swift         the per-frame pipeline (section 7)
    Geometry.swift             homography, radial model, grid tables
    Detector.swift             finder scan, orientation, k1 sweep
    Sampler.swift              luma sampling + local threshold
    Header.swift               header pack/parse (pack needed for tests)
    Fountain.swift             mulberry32, soliton, blockIndices, peeler
    Container.swift            HGC1 wrap/unwrap
    Meta.swift                 rxmeta.slim.json loading (Codable)
    CRS2/
      crs2.c                   COPY of src/crs2.c, unmodified
      crs2.h                   header you write (declarations below)
    Resources/
      rxmeta.slim.json         COPY of demo/webtxweb/rxmeta.slim.json
  HeliogramTests/
    G1_RSVectors.swift
    G2_Header.swift
    G3_Fountain.swift
    G4_Loopback.swift
    G5_OpticsVariants.swift
    Fixtures/                  COPIES of ios/fixtures/*.json
  fixtures/                    (already exists; source of truth)
  FIELDLOG.md                  you create; every field run appended
```

`project.yml` (XcodeGen; `brew install xcodegen`, then `xcodegen generate`):

```yaml
name: Heliogram
options:
  bundleIdPrefix: com.ofitzharding
  deploymentTarget:
    iOS: "17.0"
targets:
  Heliogram:
    type: application
    platform: iOS
    sources: [Heliogram]
    resources: [Heliogram/Resources]
    settings:
      base:
        SWIFT_OBJC_BRIDGING_HEADER: Heliogram/CRS2/crs2.h
        INFOPLIST_KEY_NSCameraUsageDescription: "Reads the code on the sender's screen."
        INFOPLIST_KEY_UILaunchScreen_Generation: true
        TARGETED_DEVICE_FAMILY: "1"
  HeliogramTests:
    type: bundle.unit-test
    platform: iOS
    sources: [HeliogramTests]
    dependencies: [{ target: Heliogram }]
```

`crs2.h` — the C core's public surface (implementations already exist in
crs2.c; do NOT reimplement them in Swift):

```c
#include <stdint.h>
uint32_t crc32_c(const uint8_t *buf, int len);
int certify_codeword(const uint8_t *chunk, const int32_t *order,
                     int ecc, int sub_size, int use_ladder,
                     uint8_t *block, uint8_t *coded);
```
`chunk`: 255 sampled bytes. `order`: 255 byte indices, least confident
first, or NULL for hard-only. Returns 1 and fills `block[203]` on success.
The erasure ladder (nEr = 4,10,...,<=0.7*ecc) is inside the C. This is the
hot path; at native speed one call is microseconds, so unlike the browser
you do NOT need a per-frame ladder time budget.

For the HEADER (n=68, nsym=40) you also need raw RS decode. Add to crs2.c a
thin wrapper (the ONLY allowed change, appended at the end):

```c
int rs_correct(uint8_t *msg, int n, int nsym,
               const int32_t *erase, int n_erase);
/* body: init_tables(); return correct_msg(msg, n, nsym, erase, n_erase); */
```

## 5. Capture configuration (CaptureController.swift)

- `AVCaptureSession`, `.builtInWideAngleCamera`, back position.
- Format selection: iterate `device.formats`, pick the one with
  `3840x2160` dimensions whose `videoSupportedFrameRateRanges` includes 60
  fps; set `activeFormat`, then `activeVideoMinFrameDuration =
  activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)`.
  If no 4K60 format exists on the device, fall back 4K30 and log it; the
  ceiling halves but everything still works.
- Output: `AVCaptureVideoDataOutput` with
  `kCVPixelFormatType_420YpCbCr8BiPlanarFullRange`,
  `alwaysDiscardsLateVideoFrames = true`. You read ONLY plane 0 (luma).
  Respect `CVPixelBufferGetBytesPerRowOfPlane(buf, 0)` — it is wider than
  the width; index with the stride, never with width.
- **Exposure discipline** (this kills the failure mode that ate the browser
  all night — blended and washed frames):
  1. Start in `.continuousAutoExposure` while the sender shows its white
     countdown; user taps to set `exposurePointOfInterest` on the code.
  2. On decode start (first certified header), read
     `device.exposureDuration` and `device.iso`, then switch to
     `setExposureModeCustom(duration: d, iso: i)` where
     `d = min(currentDuration, CMTime(1, 1000))` and
     `i = clamp(currentISO * currentDuration.seconds / d.seconds,
                format.minISO, format.maxISO)`.
     Short duration prevents two-seq blends; the ISO compensation keeps
     the metering.
  3. `focusMode = .locked` after initial convergence;
     `whiteBalanceMode = .locked`.
- `UIApplication.shared.isIdleTimerDisabled = true` while running.
- Threading: capture delegate writes the latest CVPixelBuffer into a
  single-slot mailbox (lock + retained buffer); a dedicated decode Thread
  loops on the mailbox. Latest-wins, no queue. UI observes an
  `@Observable` stats object on the main actor, updated at most 10 Hz.

## 6. Geometry (port from rx.html, names kept)

- `applyH`: 3x3 homography application with projective divide.
- `homography(src4, dst4)`: DLT via 8x8 Gaussian elimination, partial
  pivoting (port verbatim; it is 30 lines).
- Radial model: sample point `p' = c + (p - c) * (1 + k1 * r2 / rn2)` with
  `c = (w/2, h/2)`, `rn2 = (w*w + h*h)/4`, `r2 = |p - c|^2`.
  **Undistort-fit-redistort law**: homographies are ALWAYS fit on
  UNDISTORTED corner points (`p_u = c + (p - c)/(1 + k1*r2/rn2)`), and k1
  is applied when sampling. Fitting on raw corners and also bending samples
  double-counts the lens and fails (measured; selftest E caught it).
- Grid table (`buildGrid`): for the current (H, k1), precompute per-cell
  integer sample coordinates and a bounding box with 18 px margin. Rebuild
  only when H or k1 changes.

## 7. The per-frame decode pipeline (DecodeEngine.swift)

State: `lastH`, `K1`, `trackMiss`, `lockMiss`, grid table `SG`,
`seqAnchors`, `staged`, fountain, `dec.inferred` (set of staged-origin
symbol indices). Port the shipped rx.html logic:

1. **Tracked attempt** (if `lastH != nil && trackMiss < 8`): sample all
   cells through the grid table (bbox luma only), local threshold, try
   header (all 8 masks, hard then erasure ladder using per-byte min cell
   confidence). Header ok -> harvest + anchor.
2. **Header miss**: run the FULL codeword decode anyway.
   - If a clock fit exists (>= 3 anchors spanning >= 900 ms, fresh < 4 s)
     and the fractional predicted seq is within 0.35 of an integer:
     decode all 31 codewords, stage certified blocks under the inferred
     seq in a quarantine. One certified codeword = geometry proven,
     `trackMiss = 0`. Zero certified = `trackMiss += 1`.
   - Else if >= 1 codeword certifies (sampled bands, WITH a 2-rung
     erasure ladder — hard-only reads 0 on real glass, measured):
     keep the lock, skip the frame.
   - Else: corner refresh — re-locate each finder by normalized
     correlation against the 7x7 finder template (+1 dark ring, -1 light
     ring, +1 3x3 core, Chebyshev radii 3/2/<=1) in a +-12 px window,
     step 2, accept a corner at nscore > 0.45, need 3 of 4; refit
     (undistort-fit-redistort) and CERTIFY at the new geometry before
     installing it (uncertified refresh poisoned locks in the field).
     Failing all: `trackMiss += 1`, drop `lastH` after `lockMiss > 10`.
3. **Detection** (no lock or `trackMiss >= 8`): finder scan on the full
   luma plane — Otsu threshold (return the MIDPOINT BETWEEN CLASS MEANS,
   not the argmax bin: on two-level images argmax degenerates, measured),
   1:1:3:1:1 run scan by rows (row step = h/500), tolerances 0.6u per unit
   and 0.45*3u for the core, vertical cross-check at each candidate,
   cluster by proximity (radius 6u), take the four EXTREMES of the
   candidate cloud by (x+y, x-y). NO aspect gate. Try all 4 orientations
   of the quad (aspect-prefiltered, header-adjudicated). If nothing
   certifies at current K1, sweep
   `K1 in [.005,.01,.015,.02,.03,.045,-.005,-.01,-.02,-.03]`,
   homography refit per candidate, hard header decode as judge, at most
   one sweep per 800 ms. On a certified detection, run the +-0.005 k1
   refinement judged by certified-codeword count (same frame, same quad).
4. **Sampling** (`Sampler.swift`): per cell, mean of a 2x2 luma block at
   the (radially corrected) grid position. Local threshold: 15x15 CELL
   neighborhood mean via integral image over the cell-space luma;
   bit = lum > mean, confidence = |lum - mean|. This local threshold is
   worth 2x yield over global Otsu (measured 19.8% -> 39.2%).
5. **Certify**: assemble 255 bytes per codeword from `payload_cells`
   (MSB-first), byte confidence = min of its 8 cell confidences,
   `order` = argsort ascending, call `certify_codeword(..., use_ladder=1)`.
   Skip any `idx` already in the fountain (repeats are the common case).
   Ghost defenses, both mandatory: (a) if this seq == prevSeq+1 and the
   same j certified IDENTICAL bytes last frame, drop (rolling-shutter
   ghost; content collision odds are 2^-1600); (b) on fountain insert, if
   pool[idx - n_sub] has identical content, drop.
6. **Anchors and quarantine**: certified header -> `anchors.append((t,
   seq))` (keep 8) and, if the linear fit predicts this seq within 0.5,
   flush the quarantine into the fountain, tagging every flushed idx in
   `dec.inferred`; otherwise discard the quarantine.
7. **Close**: when decoded source count reaches k: assemble, unwrap
   container, sha256. On mismatch WITH inferred symbols present: purge all
   inferred symbols, rebuild the fountain from the untagged pool, keep
   receiving (do not stop, do not report failure).
8. **The yield engine — this is where the browser lost, build it early.**
   Final browser field state: eye (contrast margin) 74, tracking 82%,
   geometry and k1 converged, yield 0%. Diagnosis: the header RS(68,28)
   tolerates ~29% byte corruption, the payload RS(255,207) only ~9-13%,
   and the live channel sits between the two cliffs (cell BER ~2-4%
   byte-packs to 15-25%). Headers certify, payload never does. Crossing
   the payload cliff needs cell BER under ~1.5%, and these are the
   measured levers, in order:
   a. **Density**: at 4K60 with the code filling the frame you get
      12-13 native px/cell (the portrait browser runs sat at ~10.6).
      BER falls steeply with px/cell; this lever is free, coach the
      operator with the HUD px/cell readout.
   b. **Sampling**: bilinear subpixel taps at the exact (radially
      corrected) position, 3x3 weighted window, instead of the browser's
      2x2 box at integer offsets. Integer quantization at ~10 px/cell
      contributes structured, geometry-correlated errors.
   c. **CAG-lite offsets**: for codewords failing the ladder, retry with
      the whole-codeword grid shifted (dx, dy) in {-0.3, 0, +0.3} cells
      (8 neighbors); 255 resamples + one C call each. Film pipeline
      measured the family at 1.22x codewords in isolation. Cap ~10
      codewords/frame; at native cost this runs every frame.
   d. **Certified-donor equalizer** (port of tile-PRML, the film
      decoder's biggest yield machine): fit a 3x3 luma kernel on cells
      of codewords that CERTIFIED this frame (proven labels only; never
      fit on rescued observations), deconvolve the failed codewords'
      samples with it, retry. Do this after a-c if yield is still short.

## 8. Test gates (XCTest; simulator; camera not required; run BEFORE any device work)

- **G1 RS vectors**: for each of the 40 `rs_vectors` in rxmeta.slim.json
  (`chunk` b64, `order` array-or-null, `expect` b64-or-null): call
  `certify_codeword`; result must equal `expect` (null = must fail).
  40/40 or stop.
- **G2 header**: (a) pack each tuple in `header_vectors.json` and compare
  hex-identical; (b) parse each vector back and recover the fields; (c)
  corrupt 10 random bytes of a packed header, parse must still succeed
  (that is within RS(68,28)'s hard budget of 20).
- **G3 fountain**: mulberry32(42) first 8 doubles bit-identical;
  `cum` probes bit-identical (`Double.bitPattern`); `blockIndices` for
  k=997 seqs 0..39 and k=5166 seq 7 identical arrays.
- **G4 loopback**: Swift-side sender (pack header + fountain-encode +
  render cells including pseudorandom fill of unused cells, seed
  mulberry32(0x5EED1), threshold 0.5) -> synthetic luma at 12 px/cell with
  gaussian noise sigma 4, levels 20/230, margin 40 px -> the REAL pipeline
  (detection, sampling, certify, fountain) -> 100 KB random payload closes
  and sha256 matches. Mirror of the browser's selftest A+B.
- **G5 variants**: G4 rotated 90 degrees (must decode via orientation
  handling) and G4 with barrel warp kc=0.02 around the image centre (the
  k1 sweep must land near -0.02 and close). Mirror of selftest D+E.

Run: `xcodegen generate && xcodebuild test -scheme Heliogram -destination
'platform=iOS Simulator,name=iPhone 16'` (adjust to an installed sim).

## 9. UI (ReceiverView.swift) — copy the web HUD, it is field-tuned

Preview (`AVCaptureVideoPreviewLayer`, aspect FIT — cover crops the code),
guide line (state + tip + native cam-px/cell + yield% + k1), progress bar
driven by POOL FILL not decoded count (LT peeling is back-loaded; decoded
sits near 0 then avalanches — showing it made healthy transfers look
stuck), stats block: `frames / located / tracked / certified`,
`pool x/k`, `KB/s in` (2 s window over pool growth * 203/1024),
`ms/frame`, elapsed. Verdict panel: BIT-EXACT + size + full-span KB/s +
sha256 prefix + share sheet (`ShareLink`) for the recovered file.
Guide states: STRONG (>60% yield) green, DECODING (>30%) yellow, WEAK,
HOLD, STRADDLE, NO LOCK orange. px/cell guidance: <7 "MOVE CLOSER",
>16 "back off".

## 10. Field protocol (after all gates green)

1. Laptop: `python3 -m http.server 8000` in the repo,
   open `http://localhost:8000/hgtx.html?rate=56`, drop the file, verify
   HUD says `56seq/s`, F for fullscreen, SCREEN BRIGHTNESS TO MAX.
2. App: start camera, aim during the white countdown, tap the code to
   focus/meter, fill the frame (target >= 10 native px/cell on the HUD),
   hold as still as possible.
3. Record in `ios/FIELDLOG.md`: date, file size, full-span KB/s, yield,
   ms/frame, located/tracked/frames, k1, and one line on conditions.
   Every run, including failures. Failures with numbers are findings.

## 11. Measured law from the field (do not relearn these)

| law | evidence |
|---|---|
| A blended (two-seq) frame is unrecoverable; prevent, don't cure | oracle SIC left d' 0.04-1.20; 0/20 headers on 50/50 blends |
| Never integer-lock seq rate to capture rate | two field runs died parked on the seam |
| Corner pilots alone are biased; k1 chosen by certified evidence | +0.0075 vs +0.0238, 87.3% of codewords lost; live sweep chose -0.020 matching model |
| The header is one contiguous band and dies as a unit under rolling-shutter shear; payload bands degrade gracefully | field: header dead for seconds at 43% payload yield. Hence anchors + quarantine |
| Hard-only RS judges read ~0 on real glass; judges need the erasure ladder | field: certCount 0 on frames with 43% ladder yield |
| Local threshold >> global Otsu | 19.8% -> 39.2% yield |
| Interleaving hurts on this channel class | 38.1% -> 5.0%, 7.7x loss |
| Below ~8 native px/cell the payload starves | 350x194 grid: 0/32 despite verified geometry; half-res field test collapsed |
| Progress = pool fill, never decoded count | peeling is back-loaded by construction |
| The channel lives between the header's and the payload's correction cliffs | final browser run: eye 74, tracked 82%, yield 0%. Header RS(68,28) ~29% byte tolerance vs payload RS(255,207) ~9-13%. The yield engine (7.8) exists to push cell BER under the payload cliff |

## 12. Working agreements for the executing agent

- Commit after every green gate; message states the gate and the numbers.
- Never write "works" without the test or measurement in the same message.
- If a gate fails, fix forward within `ios/`; do not weaken a gate, do not
  edit fixtures. Fixtures are generated artifacts; if you believe one is
  wrong, STOP and say so with evidence.
- The wire format section (3) is frozen. Any interop bug is in your port
  by default; prove otherwise against rx.html and the fixtures.
- Sender-side brightness: when a future iOS SENDER is built,
  `UIScreen.brightness = 1.0` at transmit start. (The Mac browser sender
  cannot set brightness; it holds a wake lock and instructs the user.)
```
