"""Regular Gallager LDPC over GF(2), numpy-only.

Why this exists: hard-decision RS confines the receiver to densities where
raw BER < 1.2%, which caps information at ~291 KB/s and wastes the 330-371
KB/s regime at 9-10 px/cell (Findings §17/§19). Operating there needs a
bit-level soft code. LDPC-in-screen-camera is NOT novel (PixNet, 2010); it
is the standard vehicle. The novel part is upstream: the certified-label
channel model that makes soft metrics available at these densities at all.

Construction: column weight wc=3, row weight wr, Gallager-style stacked
random permutations. Encoding via one-time Gaussian elimination, cached to
disk (systematic form). Decoding: normalized min-sum.
"""
import hashlib
from pathlib import Path

import numpy as np

_CACHE = Path("/private/tmp/claude-502/-Users-oscarfitzharding-Documents-claude-obsidian/"
              "70e6cc8f-bc18-4a90-9409-a0b49bb559f9/scratchpad/ldpc_cache")


def _gallager_H(n, wr, wc, seed):
    assert n % wr == 0
    m_layer = n // wr
    rs = np.random.RandomState(seed)
    rows = []
    for layer in range(wc):
        perm = rs.permutation(n)
        for r in range(m_layer):
            cols = perm[r * wr:(r + 1) * wr]
            row = np.zeros(n, np.uint8)
            row[cols] = 1
            rows.append(row)
    return np.array(rows, np.uint8)


def _gauss_systematic(H):
    """Row-reduce H to [P | I_m] up to column permutation; return (Hs, colperm,
    k). Systematic encoding then fills info bits and solves parity directly."""
    Hs = H.copy()
    m, n = Hs.shape
    colperm = np.arange(n)
    r = 0
    for c in range(n - 1, -1, -1):        # pivot from the right
        if r >= m:
            break
        piv = np.flatnonzero(Hs[r:, colperm[c]]) + r
        if len(piv) == 0:
            continue
        p = piv[0]
        if p != r:
            Hs[[r, p]] = Hs[[p, r]]
        mask = Hs[:, colperm[c]].astype(bool).copy()
        mask[r] = False
        Hs[mask] ^= Hs[r]
        # move this pivot column into position n-1-r
        tgt = n - 1 - r
        colperm[[c, tgt]] = colperm[[tgt, c]]
        r += 1
    return Hs, colperm, n - r


class LDPC:
    def __init__(self, n=6000, wr=12, wc=3, seed=1):
        self.n, self.wr, self.wc = n, wr, wc
        key = hashlib.sha1(f"{n}-{wr}-{wc}-{seed}".encode()).hexdigest()[:16]
        _CACHE.mkdir(parents=True, exist_ok=True)
        f = _CACHE / f"ldpc_{key}.npz"
        if f.exists():
            z = np.load(f)
            self.H, self.Hs, self.colperm, self.k = (z["H"], z["Hs"],
                                                     z["colperm"], int(z["k"]))
        else:
            self.H = _gallager_H(n, wr, wc, seed)
            self.Hs, self.colperm, self.k = _gauss_systematic(self.H)
            np.savez_compressed(f, H=self.H, Hs=self.Hs,
                                colperm=self.colperm, k=self.k)
        m = self.H.shape[0]
        self.rank = n - self.k
        # adjacency for min-sum
        self.ci = [np.flatnonzero(self.H[i]) for i in range(m)]
        self.rate = self.k / self.n

    def encode(self, info_bits):
        """info_bits: (k,) -> codeword (n,) in natural column order."""
        assert len(info_bits) == self.k
        cw_p = np.zeros(self.n, np.uint8)          # permuted order
        cw_p[self.colperm[:self.k]] = info_bits    # info positions
        # parity: rows of Hs are [stuff | I] under colperm; solve bottom-up
        m = self.H.shape[0]
        for r in range(self.rank - 1, -1, -1):
            c = self.colperm[self.n - 1 - r]
            row = np.flatnonzero(self.Hs[r])
            acc = 0
            for cc in row:
                if cc != c:
                    acc ^= cw_p[cc]
            cw_p[c] = acc
        assert not self.syndrome(cw_p).any()
        return cw_p

    def syndrome(self, cw):
        return (self.H @ cw) & 1

    def decode(self, llr, iters=60, alpha=0.8):
        """Normalized min-sum. llr>0 means bit=0. Returns (bits, ok)."""
        m, n = self.H.shape
        Hb = self.H.astype(bool)
        M = np.zeros((m, n))                      # check->var messages
        for _ in range(iters):
            tot = llr + M.sum(axis=0)
            V = np.where(Hb, tot[None, :] - M, 0.0)   # var->check
            # per row: min |V| excluding self, and sign product
            absV = np.where(Hb, np.abs(V), np.inf)
            sgn = np.where(V < 0, -1.0, 1.0)
            rows_sign = np.where(Hb, sgn, 1.0).prod(axis=1)
            mn1 = absV.min(axis=1)
            idx1 = absV.argmin(axis=1)
            absV2 = absV.copy()
            absV2[np.arange(m), idx1] = np.inf
            mn2 = absV2.min(axis=1)
            use = np.where(np.arange(n)[None, :] == idx1[:, None],
                           mn2[:, None], mn1[:, None])
            M = np.where(Hb, alpha * rows_sign[:, None] * sgn * use, 0.0)
            hard = ((llr + M.sum(axis=0)) < 0).astype(np.uint8)
            if not self.syndrome(hard).any():
                return hard, True
        return hard, False

    def info(self, cw):
        return cw[self.colperm[:self.k]]


if __name__ == "__main__":
    # self-test: hard BSC and soft channels at the BERs that matter
    for wr in (15, 12, 10):
        code = LDPC(n=6000, wr=wr)
        rs = np.random.RandomState(7)
        info = rs.randint(0, 2, code.k).astype(np.uint8)
        cw = code.encode(info)
        print(f"(3,{wr}) n=6000  rate {code.rate:.3f}")
        for p in (0.02, 0.03, 0.05, 0.08):
            ok_n = 0
            T = 4
            for t in range(T):
                flips = rs.random(code.n) < p
                rx = cw ^ flips.astype(np.uint8)
                # soft: correct magnitude for a BSC
                L0 = np.log((1 - p) / p)
                llr = np.where(rx == 0, L0, -L0)
                dec, ok = code.decode(llr)
                ok_n += int(ok and (code.info(dec) == info).all())
            print(f"    BSC p={p:.2f}: {ok_n}/{T} blocks clean")
