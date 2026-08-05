#!/usr/bin/env python3
"""
make_web_rx.py — generate everything the BROWSER receiver needs.

The browser must agree with the transmitter on every deterministic
artifact: fountain block composition (numpy PCG64: not portable), header
whitening masks (LFSR), cell orderings, finder geometry. Rather than port
PRNGs bit-for-bit to JS, this dumps them all as data into rxmeta.json.
The page then does only signal processing, RS, CRC and the fountain,
and verifies its RS port against test vectors baked in here.

    python3 src/make_web_rx.py            # builds demo/webtxweb + rxmeta
"""
import base64
import hashlib
import json
import random
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from codec import fountain, grid

GRID = "320x208"           # 16 cam-px/cell at 1080p full-frame; the
                          # 216x140 grid needed the phone closer than
                          # is practical (measured 3 px/cell at arm
                          # length, where the payload cannot survive)
PAYLOAD = Path(__file__).parent.parent / "demo" / "kitten.png"
OUT = Path(__file__).parent.parent / "demo" / "webtxweb"
FRAMES = 1200


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=str(PAYLOAD),
                    help="ANY file to send: this becomes the transmission")
    a = ap.parse_args()
    globals()["PAYLOAD"] = Path(a.payload).expanduser()
    # 1. the sender assets (reuses the production transmitter generator)
    subprocess.run([sys.executable, str(Path(__file__).parent /
                                        "make_web_tx.py"),
                    "--payload", str(PAYLOAD), "--grid", GRID,
                    "--out", str(OUT), "--frames", str(FRAMES)], check=True)

    grid.set_ecc(48); grid.set_header_len(28); grid.set_header_centered(True)
    gw, gh = (int(v) for v in GRID.split("x"))
    L = grid.Layout(gw, gh)
    n_sub = grid.sub_count(L, grid.MODE_MONO)
    SUB = (255 - 48) - 4
    data = PAYLOAD.read_bytes()
    enc = fountain.Encoder(data, SUB)

    # 2. per-seq fountain indices (sidesteps porting PCG64)
    indices = [[int(i) for i in fountain.block_indices(s, enc.k, enc.pmf)]
               for s in range(FRAMES * n_sub // n_sub and FRAMES)]
    # every codeword j of frame seq is fountain symbol seq*n_sub+j, and its
    # composition depends on that flat index:
    flat = [[int(i) for i in fountain.block_indices(idx, enc.k, enc.pmf)]
            for idx in range(FRAMES * n_sub)]

    # 3. header whitening masks, all 8 phases, at the encoded header length
    hdr_len = len(grid.pack_header(0, enc.k, SUB, len(data),
                                   grid.MODE_MONO, 0, 0))
    masks = [base64.b64encode(grid._hdr_mask(hdr_len, ph)).decode()
             for ph in range(grid.HDR_PHASES)]

    # 4. geometry: cell orders as flat row-major indices (r*gw+c)
    pc = L.payload_cells
    hc = L.header_cells
    # Finder centre measured from the RENDERED matrix, not assumed: the
    # 1:1:3:1:1 spans cells 1..7, so the centre is cell 4 -> 4.5 in
    # cell+0.5 coordinates. f/2 = 3.5 is one cell off in all four corners,
    # which is enough to wreck the homography and kill every header.
    fc = L.finder / 2.0 + 1.0
    finders = [[fc, fc], [gw - fc, fc], [fc, gh - fc], [gw - fc, gh - fc]]

    # 5. RS test vectors so the JS port proves itself in-page
    rng = np.random.default_rng(5)
    random.seed(5)
    try:
        from creedsolo import RSCodec
    except ImportError:
        from reedsolo import RSCodec
    rs = RSCodec(48)
    vectors = []
    for _ in range(40):
        payload = bytes(rng.integers(0, 256, SUB, dtype=np.uint8))
        msg = struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF) + payload
        encd = bytes(rs.encode(msg))
        bad = bytearray(encd)
        n_err = random.randint(0, 40)
        pos = random.sample(range(255), n_err)
        for p_ in pos:
            bad[p_] ^= random.randint(1, 255)
        conf = rng.random(255)
        for p_ in pos:
            conf[p_] *= random.random()
        order = [int(i) for i in np.argsort(conf)]
        vectors.append(dict(
            chunk=base64.b64encode(bytes(bad)).decode(),
            order=order,
            block=base64.b64encode(payload).decode()
            if True else None))
    # ground truth for each vector via the python path
    import importlib
    sd = importlib.import_module("softdec")
    truth = []
    for v in vectors:
        chunk = base64.b64decode(v["chunk"])
        order = np.array(v["order"])
        # replicate certify's inner loop
        dec = None
        from reedsolo import ReedSolomonError
        try:
            dec = bytes(rs.decode(chunk)[0])
        except Exception:
            for n_er in range(4, int(48 * 0.7) + 1, 6):
                try:
                    dec = bytes(rs.decode(chunk,
                                erase_pos=[int(i) for i in order[:n_er]])[0])
                    break
                except Exception:
                    continue
        ok = (dec is not None and len(dec) >= 4 + SUB and
              zlib.crc32(dec[4:4+SUB]) & 0xFFFFFFFF ==
              struct.unpack("<I", dec[:4])[0])
        truth.append(base64.b64encode(dec[4:4+SUB]).decode() if ok else None)
    for v, t in zip(vectors, truth):
        v["expect"] = t

    # 6. STRUCTURE MAP so a JS sender can render frames without porting the
    # python renderer: which cells are finder/separator/ring, and their
    # values. Everything else is header or payload, whose order is already
    # exported above. A JS sender and JS receiver then agree with EACH
    # OTHER by construction, which is all that is required.
    import cv2 as _cv2
    blank = grid.render_frame(
        L, grid.pack_header(0, enc.k, SUB, len(data), grid.MODE_MONO, 0, 0),
        b"\x00" * L.payload_capacity_bytes(grid.MODE_MONO),
        grid.MODE_MONO, cell_px=1)
    bg = (_cv2.cvtColor(blank, _cv2.COLOR_BGR2GRAY) > 127).astype(np.uint8)
    smask = (L.is_finder | L.is_sep | L.is_ring)
    scells = [int(r) * gw + int(c) for r, c in np.argwhere(smask)]
    svals = np.packbits([int(bg[r, c]) for r, c in np.argwhere(smask)])

    meta = dict(
        struct_cells=scells,
        struct_vals=base64.b64encode(svals.tobytes()).decode(),
        gw=gw, gh=gh, n_sub=n_sub, sub=SUB, ecc=48, hdr_ecc=grid.HEADER_ECC,
        hdr_len_pre=grid.HEADER_LEN, hdr_len_enc=hdr_len,
        hdr_phases=grid.HDR_PHASES, k=enc.k, file_size=len(data),
        file_name=PAYLOAD.name,
        sha256=hashlib.sha256(data).hexdigest(),
        frames=FRAMES,
        payload_cells=[int(r) * gw + int(c) for r, c in pc],
        header_cells=[int(r) * gw + int(c) for r, c in hc],
        finders=finders,
        hdr_masks=masks,
        indices=flat,
        rs_vectors=vectors,
    )
    (OUT / "rxmeta.json").write_text(json.dumps(meta))
    sz = (OUT / "rxmeta.json").stat().st_size
    print(f"wrote {OUT}/rxmeta.json ({sz/1e6:.1f} MB), "
          f"{n_sub} cw/frame, k={enc.k}")


if __name__ == "__main__":
    main()
