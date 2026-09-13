"""tscodec: a lossless columnar codec for numeric NumPy arrays.

Public API:
    encode(array) -> bytes
    decode(buf) -> np.ndarray
"""

from .codec import encode, decode

__version__ = "0.0.0"

__all__ = ["encode", "decode", "__version__"]
