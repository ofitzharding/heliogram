# Prompt for a research LLM

Paste everything below the line.

---

You are an expert in optical wireless communication, coding theory, and
computational imaging. I am building a screen-to-camera file transfer system
and I need you to (a) find prior art that kills my claims, and (b) propose
approaches I have missed, including from unrelated fields.

**Be adversarial. Assume every novelty claim below is already published and
try to find the paper. I have already had four claims refuted this way, so
treating them as guilty until proven innocent is the correct prior.**

## The system

Display animated 2D codes full-screen on a laptop; film with a phone; recover
the file offline. No network, no pairing.

- Hardware: MacBook Pro 14" (3024x1964, 120Hz) + iPhone 17 Pro Max, 4K 60fps,
  handheld, room lights on, native camera app.
- Codec: square binary or 4-level-grey cells on a grid; four QR-style finder
  patterns; timing ring; Reed-Solomon(255, 255-48) per codeword; LT fountain
  codes across frames; CRC32 per fountain block; homography from finder
  centres plus a single radial distortion coefficient.
- Decoding is offline in Python (parallel across cores), not live.

## Measured channel constants (real captures, ground-truth verified)

These are measurements, not estimates:

- Camera blur: **sigma ~3.6 camera pixels** (fitted from finder-pattern PSF).
- **Eye opening 7.7 sigma at 13.1 camera-px/cell** (separation 156.5 luminance
  units, within-class sigma 21.9). Supports 2 levels only.
- Eye opening ~21 sigma at 17 camera-px/cell. Supports 4 levels.
- **Density wall is hard and optical.** 13.1 px/cell decodes at 0.4% bit error;
  9.1, 7.45 and 6.2 px/cell all FAIL (7.45 measured at 18% BER, errors uniform
  across the frame, true-blacks reading 119 instead of 40 — contrast collapse,
  not geometry).
- **Frame yield only ~38%**: of captures, 99% locate, 65% have readable
  headers, 59% of those decode. The comparable open-source system gets near
  100%.
- Radial distortion correction: bit error 3.73% -> 0.44% (8.5x) from one
  coefficient. Error map before correction: centre columns 0.0%, left/right
  edges 11-18%.
- Native iPhone camera (HLG/BT.2020) delivers dynamic range 205; Blackmagic
  Camera (Rec.709 SDR, manual controls) delivers 106 on identical content.

## Results

- Best REAL measured: **73.8 KB/s** goodput, bit-exact recovery of a 277 KB
  file from lights-on handheld footage (252x140 grid, binary cells).
- Best SIMULATED: 174.3 KB/s (4-level grey at 203x112 plus per-codeword
  fountain symbols). **I do not trust this number** — see "what I need" below.
- Baseline to beat: `decimen-optical-transfer` reports ~128 KB/s handheld and
  ~186 KB/s propped, decoding LIVE in a browser via zxing-wasm (QR v40).
  libcimbar reports ~106 KB/s.

## Claims already REFUTED (do not resurrect these)

1. "Fountain-bootstrapped turbo equalization is novel here" — refuted by
   Kim, Singh & Jung, *Applied Sciences* 13:9916 (2023), which promotes
   decoded symbols to virtual pilots and iterates, on the display-camera
   channel.
2. "Radial distortion correction is absent from this field" — refuted by
   libcimbar (`--undistort`, `SimpleCameraCalibration`), zxing-cpp alignment-
   pattern tiled sampling, BoofCV `setLensDistortion`.
3. "Dense mono beats sparse colour under blur is a new result" — refuted by
   Querini & Italiano on colour barcode reliability vs density.
4. "Spatially adaptive bit-loading is unexplored" — refuted by HP patent
   US2005/0254714A1 and Chen & Mow (2015).
5. "Record-then-decode is novel" — refuted by PassiveCam (2024).

## Approaches tried and MEASURED as insufficient

| approach | measured result |
|---|---|
| 2D decision-feedback ISI cancellation | 11% -> 48% BER (my implementation had a scaling bug; the idea may still be sound) |
| Transmit-side pre-equalization (inverse-filter the blur before display) | works, but cannot cross the density wall: required pre-emphasis exceeds display dynamic range |
| Receiver-side deconvolution | +0.8% accuracy only |
| Homography refinement against known cells | 18.8% -> 25.3% BER (worse) |
| PSF estimated from finder patterns + Richardson-Lucy | improves 4/6 blurred frames, rescues 0 to decodable |
| Cross-frame evidence fusion | +0.0% on handheld; channel is bimodal (frames are ~0.4% BER or 13-23%, nothing between) |
| Byte interleaving | no gain, same bimodality |
| Multi-frame super-resolution | hand tremor is 0.3-1.8 px against 11+ px cells; too small |
| Rolling-shutter sub-frame decoding | retired by information argument: code occupies ~71% of sensor pixels, so seeing 120 display frames at half each is the same total |

## What I need from you

**1. Kill or confirm my remaining claim.** Each Reed-Solomon codeword is made
its own fountain symbol, so a frame damaged in one region contributes every
codeword that survived instead of nothing (measured: 50% of frames usable ->
65% of codewords recovered from the same captures). Combined with soft
erasures placed by per-cell confidence (2e + s <= parity, so marking doubt
doubles RS reach). Is per-codeword fountain granularity with soft erasure
placement published in screen-camera, barcode, or optical camera
communication? Check libcimbar's wirehair usage specifically.

**2. Challenge the simulator.** My synthetic channel is Gaussian blur +
perspective + noise + vignette + gamma. It has been WRONG repeatedly: it
predicted 5.8 px/cell would work when reality failed at 6.2, and predicted 7
px/cell at 0.03% BER when reality gave 18%. What is it missing? Candidates I
suspect: H.265 compression on fine detail, moire between screen pixel grid and
sensor grid, display subpixel structure, sensor demosaicing, HDR tone mapping,
temporal rolling-shutter skew. Which of these dominates, and how should a
faithful screen-camera channel simulator be constructed? Cite work that has
characterised this channel properly.

**3. Capacity.** Treating the display as a band-limited 2D channel with
measured blur sigma 3.6 px, noise sigma 6, usable swing 196 levels, and
3457x1920 camera pixels of code area, a waterfilling calculation suggests
~75 KB/frame is available where I achieve ~4.3 KB/frame. Is that calculation
sound? If so, what detection architecture actually approaches it — MIMO/MMSE
equalization over the blur matrix, 2D BCJR/Viterbi, or something else? What is
the practical achievable fraction?

**4. Unrelated fields.** What transfers here from: magnetic recording (PRML,
partial-response signalling deliberately allowing ISI then using Viterbi);
adaptive optics (wavefront sensing from a guide star); radar pulse compression
(chirps plus matched filtering); CDMA (separating overlapping sources by code
orthogonality rather than physical separation); DNA data storage (constrained
codes); learned end-to-end autoencoders trained through a differentiable
channel? For each, say concretely whether the analogy holds given my measured
constants, or whether it breaks — and why.

**5. What would you do?** Given the measured constants and a hard requirement
to exceed 186 KB/s on this hardware, what is the highest-expected-value path?
Be specific and quantitative. If you believe the target is not reachable on
this hardware, say so and give the binding constraint.

Cite real papers, patents and source repositories. Where you are uncertain,
say so explicitly rather than guessing.
