#!/usr/bin/env python3
"""
container.py — what actually travels over the optical link.

The grid header carries file_SIZE but not file NAME, because until now the
payload was always the same demo PNG and the receiver knew what it was
looking at. For a real transfer the receiver has to be handed a file, not a
byte count, so the name travels inside the payload.

The sha256 travels too. Reed-Solomon plus a CRC32 per codeword makes a
corrupt block astronomically unlikely, but "astronomically unlikely" is not
"verified", and the whole point of the exercise is a transfer you can trust
without a back-channel. The receiver recomputes it and says so.

    SCF1 | name_len u16 | name utf-8 | size u64 | sha256 32B | data
"""
import hashlib
import struct

MAGIC = b"SCF1"
HDR = len(MAGIC) + 2 + 8 + 32


def wrap(name: str, data: bytes) -> bytes:
    nb = name.encode("utf-8")
    if len(nb) > 65535:
        raise ValueError("filename too long")
    return (MAGIC + struct.pack("<H", len(nb)) + nb +
            struct.pack("<Q", len(data)) + hashlib.sha256(data).digest() + data)


def unwrap(blob: bytes):
    """Returns (name, data, sha_ok). Raises ValueError if not a container."""
    if len(blob) < HDR or blob[:4] != MAGIC:
        raise ValueError("not an SCF1 container")
    p = 4
    (nlen,) = struct.unpack("<H", blob[p:p + 2]); p += 2
    name = blob[p:p + nlen].decode("utf-8", "replace"); p += nlen
    (size,) = struct.unpack("<Q", blob[p:p + 8]); p += 8
    want = blob[p:p + 32]; p += 32
    data = blob[p:p + size]
    if len(data) != size:
        raise ValueError(f"truncated: {len(data)} of {size} bytes")
    return name, data, hashlib.sha256(data).digest() == want
