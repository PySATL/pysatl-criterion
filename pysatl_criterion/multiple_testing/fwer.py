from pysatl_criterion.multiple_testing.abstract_multiple_testing import (
    AbstractMultipleTesting,
)


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
