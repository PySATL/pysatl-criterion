import json
import struct

import numpy as np
import zstandard as zstd
from sqlalchemy.types import LargeBinary, TypeDecorator


class CompressedFloatArray(TypeDecorator):
    """
    SQLAlchemy column type that stores a list of floats compressed with zstd.
    """

    impl = LargeBinary(length=2**31 - 1)
    cache_ok = True

    VERSION = 1
    DTYPE_FLOAT32 = 1
    DTYPE_FLOAT64 = 2

    def __init__(self, use_float32: bool = False):
        """
        :param use_float32: Store floats as float32 (4 bytes) instead of float64 (8 bytes)
        """
        super().__init__()
        self.use_float32 = use_float32

    def process_bind_param(self, value: list[float] | None, dialect):
        """
        Convert Python list[float] -> compressed binary for DB storage
        """
        if value is None:
            return None

        # Allow migration from old JSON/text columns
        if isinstance(value, str):
            value = json.loads(value)

        dtype = np.float32 if self.use_float32 else np.float64
        dtype_code = self.DTYPE_FLOAT32 if self.use_float32 else self.DTYPE_FLOAT64

        with np.errstate(over="ignore", invalid="ignore"):
            arr = np.asarray(value, dtype=dtype)
        if arr.ndim != 1 or not np.isfinite(arr).all():
            raise ValueError("Statistics must be a one-dimensional array of finite values")
        # SQLAlchemy copies column types across sessions; zstd contexts cannot be shared
        # by concurrent operations. Each bind/result conversion owns its context.
        compressed = zstd.ZstdCompressor().compress(arr.tobytes())

        # header: version(1B), dtype_code(1B), length(4B)
        header = struct.pack("<BBI", self.VERSION, dtype_code, len(arr))
        return header + compressed

    def process_result_value(self, value: bytes | None, dialect) -> list[float] | None:
        """
        Convert compressed binary -> Python list[float]
        """
        if value is None:
            return None

        if isinstance(value, str):
            # fallback for old DB columns stored as JSON text
            return json.loads(value)

        if len(value) < 6:
            raise ValueError("Data too short for header — probably not compressed yet")

        # Unpack header (little-endian, no padding)
        version, dtype_code, length = struct.unpack("<BBI", value[:6])
        compressed = value[6:]

        if version != self.VERSION:
            raise ValueError(f"Unsupported version: {version}")

        if dtype_code not in (self.DTYPE_FLOAT32, self.DTYPE_FLOAT64):
            raise ValueError(f"Unsupported dtype code: {dtype_code}")
        dtype = np.float32 if dtype_code == self.DTYPE_FLOAT32 else np.float64

        decompressed_bytes = zstd.ZstdDecompressor().decompress(compressed)
        if len(decompressed_bytes) != length * np.dtype(dtype).itemsize:
            raise ValueError("Compressed statistics length does not match the header")
        arr = np.frombuffer(decompressed_bytes, dtype=dtype, count=length)
        return arr.tolist()
