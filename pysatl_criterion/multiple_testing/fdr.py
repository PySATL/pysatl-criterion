from .abstract_multiple_testing import AbstractMultipleTesting
import math

class BenjaminiYekutieli(AbstractMultipleTesting):
    """
    Adjust p-values using Benjamini-Yekutieli correction for controlling FDR.

    This method controls the false discovery rate (FDR) under arbitrary dependence.
    It uses a harmonic series correction factor.

    Steps:
    1. Compute harmonic factor c = Σ(1/i) for i=1 to n
    2. Sort p-values and calculate adjusted values: p_adjusted[i] = p[i] * n * c / rank
    3. Enforce monotonicity using step-up procedure

    Parameters
    ----------
    p_values : list[float]
        List of raw p-values between 0 and 1

    Returns
    -------
    list[float]
        Adjusted p-values in original order
    """

    @classmethod
    def adjust(cls, p_values: list[float]) -> list[float]:
        n = len(p_values)
        if n == 0:
            return []

        if n > 10000:
            c = math.log(n) + 0.5772156649015328606065120900824024310421
        else:
            c = sum(1.0 / i for i in range(1, n + 1))

        sorted_indices = sorted(range(n), key=lambda i: p_values[i])
        adjusted = [0.0] * n

        for i, idx in enumerate(sorted_indices):
            rank = i + 1
            adjusted_value = p_values[idx] * n * c / rank
            adjusted[idx] = min(1.0, adjusted_value)

        current_min = adjusted[sorted_indices[-1]]
        for i in range(n - 2, -1, -1):
            idx = sorted_indices[i]
            if adjusted[idx] > current_min:
                adjusted[idx] = current_min
            else:
                current_min = adjusted[idx]

        return adjusted