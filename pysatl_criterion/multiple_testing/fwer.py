from pysatl_criterion.multiple_testing.abstract_multiple_testing import AbstractMultipleTesting


class BonferroniMultipleTesting(AbstractMultipleTesting):
    @classmethod
    def adjust(cls, p_values):
        """Adjust p-values using Bonferroni correction.

        Parameters
        ----------
        p_values : List[float]
            List of raw p-values between 0 and 1

        Returns
        -------
        p_values_adjusted: List[float]
            List of adjusted p-values
        """

        cls._validate_p_values(p_values)
        m = len(p_values)
        p_values_adjusted = [min(p_value * m, 1.0) for p_value in p_values]
        return p_values_adjusted

    @classmethod
    def test(cls, p_values, threshold=0.05):
        """Test hypotheses using Bonferroni correction.

        Parameters
        ----------
        p_values : List[float]
            List of raw p-values
        threshold : float, optional (default is 0.05)
            Significance level for controlling FWER (Family-Wise Error Rate)

        Returns
        -------
        (rejected, p_values_adjusted): Tuple[List[bool], List[float]]
            - rejected: boolean list of rejection decisions
            - p_values_adjusted: list of adjusted p-values
        """

        return super().test(p_values, threshold)


class SidakMultipleTesting(AbstractMultipleTesting):
    @classmethod
    def adjust(cls, p_values):
        """Adjust p-values using Šidák correction.

        Parameters
        ----------
        p_values : List[float]
            List of raw p-values between 0 and 1

        Returns
        -------
        p_values_adjusted: List[float]
            List of adjusted p-values

        Notes
        -----
        The Šidák correction controls FWER under either of:
        1. Independence of all p-values
        2. Positive lower orthant dependence (PLOD)
            (i.e., P(X₁ ≤ x₁, ..., Xₙ ≤ xₙ) ≥ Π P(Xᵢ ≤ xᵢ) for all x)
        """

        cls._validate_p_values(p_values)
        m = len(p_values)
        p_values_adjusted = [1 - (1 - p_value) ** m for p_value in p_values]
        return p_values_adjusted

    @classmethod
    def test(cls, p_values, threshold=0.05):
        """Test hypotheses using Šidák correction.

        Parameters
        ----------
        p_values : List[float]
            List of raw p-values
        threshold : float, optional (default is 0.05)
            Significance level for controlling FWER (Family-Wise Error Rate)

        Returns
        -------
        (rejected, p_values_adjusted): Tuple[List[bool], List[float]]
            - rejected: boolean list of rejection decisions
            - p_values_adjusted: list of adjusted p-values

        Notes
        -----
        The Šidák correction controls FWER under either of:
        1. Independence of all p-values
        2. Positive lower orthant dependence (PLOD)
            (i.e., P(X₁ ≤ x₁, ..., Xₙ ≤ xₙ) ≥ Π P(Xᵢ ≤ xᵢ) for all x)
        """

        return super().test(p_values, threshold)

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
