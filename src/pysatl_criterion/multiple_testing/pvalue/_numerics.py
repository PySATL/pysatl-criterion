"""Stable numerical kernels shared by p-value procedures."""

import numpy as np

from pysatl_criterion.multiple_testing.inputs import FloatArray


def sidak_adjustment(p: FloatArray, counts: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        return -np.expm1(counts * np.log1p(-p))
