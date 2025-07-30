from .abstract_multiple_testing import AbstractMultipleTesting

class Holm(AbstractMultipleTesting):
    """
    Adjust p-values using the Holm-Bonferroni method for multiple testing correction.

    This method controls the family-wise error rate (FWER) and is more powerful than
    the standard Bonferroni correction. It uses a step-down procedure.

    Steps:
    1. Sort p-values and keep original indices
    2. Calculate adjusted p-values: p_adjusted[i] = p[i] * (n - i)
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

        sorted_indices = sorted(range(n), key=lambda i: p_values[i])
        adjusted = [0.0] * n

        for i, idx in enumerate(sorted_indices):
            adjusted_value = p_values[idx] * (n - i)
            adjusted[idx] = min(1.0, adjusted_value)

        for i in range(n - 2, -1, -1):
            if adjusted[sorted_indices[i]] > adjusted[sorted_indices[i + 1]]:
                adjusted[sorted_indices[i]] = adjusted[sorted_indices[i + 1]]

        return adjusted


class SidakHolm(AbstractMultipleTesting):
    @classmethod
    def adjust(cls, p_values: list[float]) -> list[float]:
        """
        Adjust p-values using the Šidák correction for multiple hypothesis testing.

        Parameters
        ----------
        p_values : list[float]
            List of raw p-values for hypothesis testing. Must be in range [0, 1].

        Returns
        -------
        list[float]
            List of adjusted p-values, each in range [0, 1].
        """
        n = len(p_values)
        if n == 0:
            return []
        return [min(1.0, 1 - (1 - p) ** n) for p in p_values]