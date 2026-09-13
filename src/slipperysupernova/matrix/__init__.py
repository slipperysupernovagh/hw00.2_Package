"""
Matrix subpackage containing matrix functions
"""
from .elementary import rowreplacement, rowscale, rowswap, rref

__all__ = [rowswap,rref,rowscale,rowreplacement]