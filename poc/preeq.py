#!/usr/bin/env python3
"""
preeq.py — transmit-side pre-equalization. Beat blur BEFORE it happens.

THE ASYMMETRY THIS EXPLOITS
---------------------------
Camera blur is a linear operator K. Every screen-camera system in existence
displays a crisp binary pattern T, the camera observes K*T, and the receiver
fights the damage. Inverting K at the RECEIVER amplifies noise — measured on
our own footage, receiver-side deconvolution bought only +0.8%.

But we control the transmitter, and there the situation is completely
different: no sensor noise, no quantization beyond 8 bits, and perfect
knowledge of what we want. So solve for the display image D such that

        (D * K) sampled at cell centres  ==  T

and display D instead of T. The camera's own blur then reconstructs crisp
symbols. Pre-emphasis, exactly as DSL pre-compensates copper.

WHY IT SHOULD MOVE THE DENSITY WALL
-----------------------------------
Density is limited by "eye opening": the gap between what a black cell and a
white cell read after blur. Measured on real footage at 6.17 px/cell, blacks
read 110 where they should read ~40 — the eye had closed. Pre-equalization
restores the opening by spending display dynamic range instead of cell area:
overshoot around each cell so the blurred result lands back at the extremes.

The cost is contrast headroom (we can't drive below 0 or above 255), so the
solver works inside a reduced nominal range and uses the freed headroom for
overshoot. Less average contrast, far better per-cell separation.

    python3 preeq.py --demo          # simulate: how far does the wall move?
"""
import argparse

import cv2
import numpy as np


def gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    if size is None:
        size = max(3, int(sigma * 6) | 1)
    k = cv2.getGaussianKernel(size, sigma)
    return (k @ k.T).astype(np.float32)


def render_cells(cells: np.ndarray, cell_px: int, lo: float, hi: float) -> np.ndarray:
    """Binary cell grid -> display image at cell_px per cell."""
    img = np.where(cells > 0, hi, lo).astype(np.float32)
    return np.kron(img, np.ones((cell_px, cell_px), np.float32))


def sample_centres(img: np.ndarray, cell_px: int) -> np.ndarray:
    """Sample at cell centres (3x3 box, as the real receiver does)."""
    box = cv2.boxFilter(img, cv2.CV_32F, (3, 3))
    off = cell_px // 2
    return box[off::cell_px, off::cell_px]


def cell_kernel(kernel: np.ndarray, cell_px: int, radius: int = 4) -> np.ndarray:
    """Effective CELL-to-CELL response: how much cell j leaks into sample i.

    Measured empirically by rendering a single-cell impulse, blurring it with
    the pixel-domain kernel, and sampling at cell centres. This collapses the
    whole pixel-domain problem to a small cell-domain convolution, which is
    what makes the inverse stable and cheap.
    """
    n = 2 * radius + 1
    imp = np.zeros((n, n), np.float32)
    imp[radius, radius] = 1.0
    big = np.kron(imp, np.ones((cell_px, cell_px), np.float32))
    blurred = cv2.filter2D(big, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return sample_centres(blurred, cell_px)[:n, :n]


def _kernel_fft(Keff: np.ndarray, H: int, W: int) -> np.ndarray:
    """Zero-phase FFT of a centred kernel, padded to (H, W).

    The kernel's centre must sit at index (0,0) before transforming, otherwise
    the convolution result comes out shifted by the kernel radius — which was
    exactly the bug that made the first two attempts produce 50% BER.
    """
    r = Keff.shape[0] // 2
    pad = np.zeros((H, W), np.float32)
    pad[:Keff.shape[0], :Keff.shape[1]] = Keff
    pad = np.roll(pad, (-r, -r), axis=(0, 1))
    return np.fft.rfft2(pad)


def forward_model(V: np.ndarray, Keff: np.ndarray) -> np.ndarray:
    """Predict sampled values from displayed cell values (cell-domain)."""
    H, W = V.shape
    return np.fft.irfft2(np.fft.rfft2(V) * _kernel_fft(Keff, H, W), s=(H, W))


def preequalize(cells: np.ndarray, kernel: np.ndarray, cell_px: int,
                headroom: float = 0.35, reg: float = 0.02):
    """Display image whose blurred, sampled version reproduces `cells`.

    Solved as a cell-domain inverse filter (regularised Wiener) rather than
    pixel-domain iteration: find cell drive values V with V (*) Keff = target,
    then render V as blocks. Because the transmitter is noiseless, the inverse
    can be applied aggressively — the only real constraint is that V must fit
    in the display's [0, 255] range, which `headroom` reserves room for.
    """
    lo = 255.0 * headroom / 2.0
    hi = 255.0 * (1.0 - headroom / 2.0)
    target = np.where(cells > 0, hi, lo).astype(np.float32)

    Keff = cell_kernel(kernel, cell_px)
    H, W = target.shape
    KF = _kernel_fft(Keff, H, W)
    mu = target.mean()
    TF = np.fft.rfft2(target - mu)
    # regularised inverse: conj(K) / (|K|^2 + reg)
    VF = TF * np.conj(KF) / (np.abs(KF) ** 2 + reg)
    V = np.fft.irfft2(VF, s=(H, W)) + mu
    np.clip(V, 0, 255, out=V)
    D = np.kron(V.astype(np.float32), np.ones((cell_px, cell_px), np.float32))
    return D, target


def eye_opening(sampled: np.ndarray, cells: np.ndarray) -> float:
    """Separation between black and white populations, in units of their spread.

    This is the quantity that actually decides whether a density works: our
    real footage decoded fine at 9.8 sigma and failed when it collapsed.
    """
    w = sampled[cells > 0]
    b = sampled[cells == 0]
    if w.size < 4 or b.size < 4:
        return 0.0
    spread = (w.std() + b.std()) / 2.0
    return float((w.mean() - b.mean()) / max(spread, 1e-6))


def ber(sampled: np.ndarray, cells: np.ndarray) -> float:
    th = (sampled.max() + sampled.min()) / 2.0
    u8 = np.clip(sampled, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    got = (sampled > t).astype(np.uint8)
    return float((got != (cells > 0).astype(np.uint8)).mean())


def measure(S, cells, noise, rng):
    """BER at the midpoint threshold, with camera noise added.

    Uses the midpoint of the two class means rather than Otsu: on synthetic
    data with near-zero class variance Otsu can land exactly on a class and
    report a spurious 50%. The real decoder sees rich histograms where Otsu is
    fine; here we want to measure the CHANNEL, not the thresholder.
    """
    Sn = S + rng.normal(0, noise, S.shape)
    w, b = Sn[cells > 0], Sn[cells == 0]
    t = (w.mean() + b.mean()) / 2.0
    got = (Sn > t).astype(np.uint8)
    spread = (w.std() + b.std()) / 2.0
    eye = (w.mean() - b.mean()) / max(spread, 1e-6)
    return float((got != (cells > 0).astype(np.uint8)).mean()), float(eye)


def demo():
    rng = np.random.default_rng(7)
    n = 120
    cells = rng.integers(0, 2, (n, n)).astype(np.uint8)

    # Blur measured on real footage: sigma such that a 12.9 px/cell code sits
    # around 0.4% BER and a 6.2 px/cell code collapses. sigma ~= 1.6 px.
    sigma = 1.6
    K = gaussian_kernel(sigma)

    print(f"blur sigma = {sigma} px  (calibrated to our measured density wall)")
    print()
    print(f"{'px/cell':>8s}  {'plain eye':>10s} {'plain BER':>10s}   "
          f"{'preeq eye':>10s} {'preeq BER':>10s}   {'verdict':>12s}")
    RS_LIMIT = 0.0123     # ECC 48 corrects up to ~1.23% bit error
    for cp in (13, 11, 9, 7, 6, 5, 4, 3):
        # plain: crisp binary at full contrast
        Dp = render_cells(cells, cp, 0.0, 255.0)
        Sp = sample_centres(cv2.filter2D(Dp, -1, K, borderType=cv2.BORDER_REPLICATE), cp)
        Sp = Sp[:n, :n]
        ep, bp = eye_opening(Sp, cells), ber(Sp, cells)

        # pre-equalized
        Dq, _ = preequalize(cells, K, cp)
        Sq = sample_centres(cv2.filter2D(Dq, -1, K, borderType=cv2.BORDER_REPLICATE), cp)
        Sq = Sq[:n, :n]
        eq, bq = eye_opening(Sq, cells), ber(Sq, cells)

        v = ""
        if bp > RS_LIMIT and bq <= RS_LIMIT:
            v = "PREEQ WINS"
        elif bq <= RS_LIMIT:
            v = "both ok"
        else:
            v = "both fail"
        print(f"{cp:8d}  {ep:10.1f} {bp*100:9.2f}%   {eq:10.1f} {bq*100:9.2f}%   {v:>12s}")

    print()
    print("Interpretation: the lowest px/cell where BER stays under 1.23% is the")
    print("density wall. Every px/cell of headroom is (13/x)^2 more data per frame.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo()
