import numpy as np


def _sum_digits(n: int, nBits: int) -> int:
    width = len(str(nBits))   # nBits is 1024 or 2048 so width = 4
    mod = np.power(10, width, dtype=np.int32)
    r = 0
    while n:
        r, n = r + n % mod, n//mod
    
    return r

def _hash_fold(nze: dict, nBits: int) -> np.array:
    # create vector of zeros as a placeholder
    vec = np.zeros(nBits, dtype=np.int32)
    bits = list(nze. keys())
    for bit in bits:
        n = _sum_digits (bit, nBits)
        idx = n % nBits
        vec[idx] = nze[bit]

    return vec
