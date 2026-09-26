import secrets
import math

def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)   # in [0, 1)
    return a + (b - a) * u

def exponentialdist(lam: float):
    y = uniform()
    x = (-1/lam)*math.log(y)
    return x