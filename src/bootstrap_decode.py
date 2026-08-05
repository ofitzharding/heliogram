#!/usr/bin/env python3
"""
bootstrap_decode.py — decision-directed, fountain-bootstrapped equalization.

The pass-7 finding: per-cell "noise" in this channel is a deterministic
fingerprint (residuals correlate 0.96 between captures of the same frame).
So the receiver can LEARN it — and the fountain code supplies the training
data, because every successfully decoded frame is a fully-known test
pattern. Loop:

    decode what you can
      -> fit a local channel model on the decoded frames
      -> re-detect every frame with per-cell learned thresholds
      -> decode more -> refit -> ...

Two-pass offline architecture: pass A samples every frame once (video is
read a single time); the bootstrap then iterates in memory.

    python3 bootstrap_decode.py capture.mov out.bin --grid 252x140 --ecc 80
"""
import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid

TILE = 10          # cells per tile side for the local channel model
ML_MARGIN = 6.0


def sample_pass(cap, layout):
    """Read the video once; return per-frame samples and header info."""
    frames = []   # dicts: n, lum(float16), hdr_lum, seq (or None), how
    n = 0
    proto = None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if n % 200 == 0:
            print(f"  pass A: frame {n}", file=sys.stderr)
        if img.shape[1] >= 3000:
            small = cv2.resize(img, None, fx=0.5, fy=0.5)
            Hs = grid.locate(small, layout)
            H = np.diag([2.0, 2.0, 1.0]) @ Hs if Hs is not None else None
        else:
            H = grid.locate(img, layout)
        if H is None:
            continue
        header, samples, st = grid.sample_frame(img, layout, H)
        hdr_lum = grid.sample_cells(img, layout, H, layout.header_cells).mean(axis=1)
        lum = samples.mean(axis=1).astype(np.float16)
        frames.append(dict(n=n, lum=lum, hdr=hdr_lum.astype(np.float16),
                           seq=None if header is None else header["seq"],
                           how="hard" if header is not None else None))
        if header is not None and proto is None:
            proto = header
    return frames, proto


def assign_seqs(frames, proto, layout):
    """Give every frame a seq: hard header, else ML template, else clock."""
    templates = grid.header_templates(proto, 2000)
    for f in frames:
        if f["seq"] is None:
            seq, margin = grid.ml_header_seq(f["hdr"].astype(np.float64), templates)
            if margin >= ML_MARGIN:
                f["seq"], f["how"] = int(seq), "ml"
    # loop period from repeated seqs
    seen = {}
    diffs = []
    for f in frames:
        if f["seq"] is not None:
            if f["seq"] in seen:
                diffs.append(f["n"] - seen[f["seq"]])
            seen[f["seq"]] = f["n"]
    period = int(np.median(diffs)) if diffs else None
    anchors = [(f["n"], f["seq"]) for f in frames if f["seq"] is not None]
    if period and anchors:
        ns = np.array([a[0] for a in anchors])
        for f in frames:
            if f["seq"] is None:
                i = int(np.argmin(np.abs(ns - f["n"])))
                n0, s0 = anchors[i]
                if abs(f["n"] - n0) <= 40:
                    f["seq"] = int((s0 + (f["n"] - n0)) % period)
                    f["how"] = "clock"
    counts = {}
    for f in frames:
        counts[f["how"]] = counts.get(f["how"], 0) + 1
    print(f"seq assignment: {counts}  (loop period ~{period} captures)")
    return period


def true_bit_field(seq, layout, enc, block_size, ecc):
    from reedsolo import RSCodec
    block = enc.block(seq)
    block = block + b"\x00" * (block_size - len(block))
    payload = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF) + block
    coded = bytes(RSCodec(ecc).encode(payload))
    tb = np.unpackbits(np.frombuffer(coded, dtype=np.uint8))
    m = min(len(tb), len(layout.payload_cells))
    F = np.full((layout.gh, layout.gw), 0.5, dtype=np.float32)
    cells = layout.payload_cells[:m]
    F[cells[:, 0], cells[:, 1]] = tb[:m]
    return F, tb[:m]


def neighbor_features(F, cells):
    fs = [np.ones(len(cells), dtype=np.float32)]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = np.clip(cells[:, 0] + dr, 0, F.shape[0] - 1)
            c = np.clip(cells[:, 1] + dc, 0, F.shape[1] - 1)
            fs.append(F[r, c])
    return np.stack(fs, axis=1)   # [:, 5] is the own cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--grid", default="252x140")
    ap.add_argument("--ecc", type=int, default=80)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--train-seconds", type=float, default=0,
                    help="steady-state protocol: bootstrap ONLY on the first "
                         "T seconds, then measure cold-start time-to-file on "
                         "the remainder with the trained model. This is the "
                         "honest goodput of a receiver that already knows the "
                         "fingerprint.")
    args = ap.parse_args()
    grid.set_ecc(args.ecc)

    gw, gh = (int(v) for v in args.grid.split("x"))
    layout = grid.Layout(gw, gh)
    t0 = time.time()

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frames, proto = sample_pass(cap, layout)
    print(f"pass A: {len(frames)} frames located, "
          f"{sum(f['seq'] is not None for f in frames)} hard headers")
    if proto is None:
        sys.exit("no readable header anywhere — cannot bootstrap")
    assign_seqs(frames, proto, layout)

    cells = layout.payload_cells
    m = len(cells)
    tile_of = ((cells[:, 0] // TILE) * (gw // TILE + 1) + cells[:, 1] // TILE)
    tiles = np.unique(tile_of)

    train_cut = None
    if args.train_seconds > 0:
        train_cut = int(args.train_seconds * fps)
        print(f"steady-state protocol: training on captures 1..{train_cut}, "
              f"measuring on the rest")

    dec = fountain.Decoder(proto["k"], proto["block_size"], proto["file_size"])
    decoded_blocks = {}   # seq -> block bytes (CRC-verified)

    def try_block(seq, bits, votes=None):
        """bits -> RS -> CRC -> fountain. votes (0..1) provide soft margins."""
        pseudo = (bits.astype(np.float32) * 255.0 if votes is None
                  else votes.astype(np.float32) * 255.0)
        payload = grid.decide_payload(dict(proto, seq=seq),
                                      pseudo[:, None].repeat(3, axis=1), layout)
        if payload is None:
            return False
        bs = proto["block_size"]
        crc = struct.unpack("<I", payload[:4])[0]
        block = payload[4:4 + bs]
        if zlib.crc32(block) & 0xFFFFFFFF != crc:
            return False
        if seq not in decoded_blocks:
            decoded_blocks[seq] = block
            dec.add(seq, block)
        return True

    # cache of true fields for decoded seqs
    fields = {}
    data_stub = None   # encoder built lazily from recovered file? No — from blocks:
    # We can regenerate a decoded frame's true bits directly from its block.
    from reedsolo import RSCodec
    def field_for(seq):
        if seq in fields:
            return fields[seq]
        block = decoded_blocks[seq]
        payload = struct.pack("<I", zlib.crc32(block) & 0xFFFFFFFF) + block
        coded = bytes(RSCodec(grid.PAYLOAD_ECC).encode(payload))
        tb = np.unpackbits(np.frombuffer(coded, dtype=np.uint8))[:m]
        F = np.full((gh, gw), 0.5, dtype=np.float32)
        F[cells[:len(tb), 0], cells[:len(tb), 1]] = tb
        fields[seq] = (F, tb)
        return fields[seq]

    # ---- round 0: plain per-frame Otsu + vote fusion (the old receiver) ----
    test_frames = None
    if train_cut is not None:
        test_frames = [f for f in frames if f["n"] > train_cut]
        frames = [f for f in frames if f["n"] <= train_cut]
    by_seq = {}
    for f in frames:
        if f["seq"] is not None:
            by_seq.setdefault(f["seq"], []).append(f)
    for seq, obs in by_seq.items():
        acc = np.zeros(m, dtype=np.float32)
        for i, f in enumerate(obs, 1):
            lum = f["lum"][:m].astype(np.float32)
            th, _ = cv2.threshold(np.clip(lum, 0, 255).astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            acc += (lum > th)
            # try after every capture: fusion depth that works varies per frame,
            # and one clean capture can beat an average polluted by bad ones
            if try_block(seq, acc / i > 0.5, acc / i):
                break
    print(f"round 0 (scanner): {len(decoded_blocks)}/{proto['k']} blocks")

    # ---- bootstrap rounds ----
    for rnd in range(1, args.rounds + 1):
        if dec.done:
            break
        train_seqs = [s for s in decoded_blocks if s in by_seq]
        if not train_seqs:
            print("no training material; stopping")
            break
        # fit per-tile linear model + per-cell offset on ALL captures of
        # decoded seqs
        X_list, y_list, t_list, cell_idx = [], [], [], []
        for s in train_seqs:
            F, tb = field_for(s)
            feats = neighbor_features(F, cells)
            for f in by_seq[s]:
                X_list.append(feats)
                y_list.append(f["lum"][:m].astype(np.float32))
                t_list.append(tile_of)
                cell_idx.append(np.arange(m))
        X = np.concatenate(X_list); y = np.concatenate(y_list)
        T = np.concatenate(t_list); CI = np.concatenate(cell_idx)
        coef = {}
        resid = np.zeros_like(y)
        for t in tiles:
            sel = T == t
            if sel.sum() < 60:
                continue
            c, _, _, _ = np.linalg.lstsq(X[sel], y[sel], rcond=None)
            coef[t] = c
            resid[sel] = y[sel] - X[sel] @ c
        # per-cell offset: mean residual per cell
        offset = np.zeros(m, dtype=np.float32)
        cnt = np.zeros(m, dtype=np.int32)
        np.add.at(offset, CI, resid)
        np.add.at(cnt, CI, 1)
        offset = np.where(cnt >= 3, offset / np.maximum(cnt, 1), 0.0)

        # per-cell threshold at neutral neighbours: c0 + 0.5*sum(coefs) + offset
        thr = np.zeros(m, dtype=np.float32)
        own_gain = np.zeros(m, dtype=np.float32)
        for t in tiles:
            if t not in coef:
                continue
            sel_c = tile_of == t
            c = coef[t]
            thr[sel_c] = c[0] + 0.5 * c[1:].sum() + offset[sel_c]
            own_gain[sel_c] = c[5]
        ok_cells = own_gain > 1.0   # cells where the model learned a real gain
        last_thr, last_ok = thr, ok_cells

        new = 0
        for seq, obs in by_seq.items():
            if seq in decoded_blocks:
                continue
            acc = np.zeros(m, dtype=np.float32)
            hit = False
            for i, f in enumerate(obs, 1):
                lum = f["lum"][:m].astype(np.float32)
                bits = np.where(ok_cells, lum > thr, lum > np.median(lum))
                acc += bits
                if try_block(seq, acc / i > 0.5, acc / i):
                    hit = True
                    break
            if hit:
                new += 1
        print(f"round {rnd} (equalized): +{new} -> "
              f"{len(decoded_blocks)}/{proto['k']} blocks "
              f"({(ok_cells).mean()*100:.0f}% cells modeled, "
              f"{len(train_seqs)} training frames)")
        if new == 0:
            break

    # ---- steady-state measurement: trained receiver, cold start on the tail ----
    if train_cut is not None:
        if "last_thr" not in dir():
            sys.exit("training phase never fit a model — lengthen --train-seconds")
        dec2 = fountain.Decoder(proto["k"], proto["block_size"], proto["file_size"])
        got2 = set()
        votes = {}
        n_start = train_cut
        n_done = None
        for f in test_frames:
            if f["seq"] is None:
                continue
            lum = f["lum"][:m].astype(np.float32)
            bits = np.where(last_ok, lum > last_thr, lum > np.median(lum))
            acc = votes.setdefault(f["seq"], [np.zeros(m, dtype=np.float32), 0])
            acc[0] += bits
            acc[1] += 1
            if f["seq"] in got2:
                continue
            pseudo = (acc[0] / acc[1] * 255.0)
            payload = grid.decide_payload(dict(proto, seq=f["seq"]),
                                          pseudo[:, None].repeat(3, axis=1), layout)
            if payload is None:
                continue
            bs = proto["block_size"]
            crc = struct.unpack("<I", payload[:4])[0]
            block = payload[4:4 + bs]
            if zlib.crc32(block) & 0xFFFFFFFF != crc:
                continue
            got2.add(f["seq"])
            dec2.add(f["seq"], block)
            if not dec2.done and len(got2) >= dec2.k:
                dec2.gaussian_fallback()
            if dec2.done:
                n_done = f["n"]
                break
        if n_done is None:
            # Tail too short to gather k blocks. Report the quantity that
            # actually determines throughput: per-capture decode yield with the
            # trained model. time-to-file = k / (fps * yield), which is the
            # honest steady-state rate, independent of how long this clip is.
            attempted = sum(1 for f in test_frames if f["seq"] is not None)
            yield_rate = len(got2) / max(1, len(set(f["seq"] for f in test_frames
                                                    if f["seq"] is not None)))
            t_file = proto["k"] / (fps * max(yield_rate, 1e-6))
            print(f"\nSTEADY-STATE (trained receiver, single-pass yield):")
            print(f"  tail: {attempted} captures, "
                  f"{len(got2)} distinct blocks recovered of {proto['k']} needed")
            print(f"  per-frame decode yield: {yield_rate*100:.0f}%")
            print(f"  => time-to-file {t_file:.2f}s at {fps:.0f} fps display")
            print(f"  => PROJECTED GOODPUT {proto['file_size']/t_file/1024:.1f} KB/s")
            print(f"     (projection from measured yield, not a timed transfer)")
            sys.exit(0)
        data = dec2.result()
        Path(args.output).write_bytes(data)
        import hashlib
        secs = (n_done - n_start) / fps
        print(f"\nSTEADY-STATE (trained receiver, cold start):")
        print(f"  recovered {len(data):,} bytes  "
              f"sha256 {hashlib.sha256(data).hexdigest()[:16]}")
        print(f"  time-to-file: {secs:.2f}s of capture "
              f"({len(got2)} blocks gathered)")
        print(f"  GOODPUT {len(data)/secs/1024:.1f} KB/s")
        return

    if not dec.done:
        dec.gaussian_fallback()
    if not dec.done:
        got = len(dec.decoded)
        sys.exit(f"FAILED: {got}/{proto['k']} after bootstrap "
                 f"({len(decoded_blocks)} verified blocks)")

    data = dec.result()
    Path(args.output).write_bytes(data)
    wall = time.time() - t0
    import hashlib
    n_capture_frames = frames[-1]["n"]
    capture_seconds = n_capture_frames / fps
    print(f"\nrecovered {len(data):,} bytes  "
          f"sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    print(f"capture: {capture_seconds:.1f}s of video")
    print(f"GOODPUT {len(data)/capture_seconds/1024:.1f} KB/s   "
          f"(decode wall time {wall:.0f}s)")


if __name__ == "__main__":
    main()
