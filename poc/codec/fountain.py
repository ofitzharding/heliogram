"""LT fountain code: encoder + peeling decoder with GF(2) elimination fallback.

Deterministic: block seq number seeds the PRNG, so transmitter and receiver
agree on each block's composition with no back-channel. This is the same
scheme prior tools use; ours additionally exposes a path for
soft-decision decoding later (see decode notes in ../decode.py).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.uint64(0x9E3779B97F4A7C15 ^ np.uint64(seed)))


def robust_soliton(k: int, c: float = 0.03, delta: float = 0.5) -> np.ndarray:
    """Degree distribution. Returns pmf over degrees 1..k."""
    s = c * math.log(k / delta) * math.sqrt(k)
    rho = np.zeros(k + 1)
    rho[1] = 1.0 / k
    for d in range(2, k + 1):
        rho[d] = 1.0 / (d * (d - 1))
    tau = np.zeros(k + 1)
    pivot = max(1, min(k, int(round(k / s)))) if s > 0 else k
    for d in range(1, pivot):
        tau[d] = s / (k * d)
    tau[pivot] = s * math.log(s / delta) / k if s > 1 else 0
    pmf = rho + tau
    pmf = np.clip(pmf, 0, None)
    return pmf[1:] / pmf[1:].sum()


def block_indices(seq: int, k: int, pmf: np.ndarray) -> np.ndarray:
    """Which source blocks XOR into encoded block `seq`. Deterministic in seq."""
    g = _rng(seq)
    d = int(g.choice(len(pmf), p=pmf)) + 1
    return g.choice(k, size=d, replace=False)


class Encoder:
    def __init__(self, data: bytes, block_size: int):
        self.block_size = block_size
        pad = (-len(data)) % block_size
        padded = data + b"\x00" * pad
        self.k = len(padded) // block_size
        self.blocks = np.frombuffer(padded, dtype=np.uint8).reshape(self.k, block_size)
        self.pmf = robust_soliton(self.k)
        self.file_size = len(data)
        self.file_hash = hashlib.sha256(data).digest()[:8]

    def block(self, seq: int) -> bytes:
        idx = block_indices(seq, self.k, self.pmf)
        out = np.bitwise_xor.reduce(self.blocks[idx], axis=0)
        return out.tobytes()


class Decoder:
    def __init__(self, k: int, block_size: int, file_size: int):
        self.k = k
        self.block_size = block_size
        self.file_size = file_size
        self.pmf = robust_soliton(k)
        self.decoded: dict[int, np.ndarray] = {}
        self.pending: list[tuple[set[int], np.ndarray]] = []
        self.seen: set[int] = set()

    @property
    def done(self) -> bool:
        return len(self.decoded) == self.k

    def add(self, seq: int, payload: bytes) -> None:
        if seq in self.seen or self.done:
            return
        self.seen.add(seq)
        idx = set(int(i) for i in block_indices(seq, self.k, self.pmf))
        data = np.frombuffer(payload, dtype=np.uint8).copy()
        # reduce against already-decoded blocks
        for i in list(idx):
            if i in self.decoded:
                data ^= self.decoded[i]
                idx.discard(i)
        if not idx:
            return
        if len(idx) == 1:
            self._resolve(idx.pop(), data)
        else:
            self.pending.append((idx, data))

    def _resolve(self, i: int, data: np.ndarray) -> None:
        """Peel: a newly decoded source block may unlock pending blocks."""
        stack = [(i, data)]
        while stack:
            i, data = stack.pop()
            if i in self.decoded:
                continue
            self.decoded[i] = data
            still = []
            for idx, pdata in self.pending:
                if i in idx:
                    pdata ^= data
                    idx.discard(i)
                if len(idx) == 1:
                    stack.append((idx.pop(), pdata))
                elif idx:
                    still.append((idx, pdata))
            self.pending = still

    def gaussian_fallback(self) -> bool:
        """GF(2) elimination over everything held. Rescues stalls near k."""
        if self.done:
            return True
        unknown = sorted(set(range(self.k)) - set(self.decoded))
        col = {b: j for j, b in enumerate(unknown)}
        rows, rhs = [], []
        for idx, data in self.pending:
            r = np.zeros(len(unknown), dtype=np.uint8)
            for i in idx:
                r[col[i]] = 1
            rows.append(r)
            rhs.append(data.copy())
        if not rows:
            return False
        A = np.array(rows, dtype=np.uint8)
        B = np.array(rhs, dtype=np.uint8)
        m, n = A.shape
        piv_of_col = {}
        r = 0
        for c in range(n):
            sel = None
            for i in range(r, m):
                if A[i, c]:
                    sel = i
                    break
            if sel is None:
                continue
            A[[r, sel]] = A[[sel, r]]
            B[[r, sel]] = B[[sel, r]]
            for i in range(m):
                if i != r and A[i, c]:
                    A[i] ^= A[r]
                    B[i] ^= B[r]
            piv_of_col[c] = r
            r += 1
        if len(piv_of_col) < n:
            return False
        for c, i in piv_of_col.items():
            self.decoded[unknown[c]] = B[i]
        self.pending = []
        return True

    def result(self) -> bytes:
        assert self.done
        out = np.concatenate([self.decoded[i] for i in range(self.k)])
        return out.tobytes()[: self.file_size]
