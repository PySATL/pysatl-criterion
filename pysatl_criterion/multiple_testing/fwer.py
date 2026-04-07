from pysatl_criterion.multiple_testing.abstract_multiple_testing import AbstractMultipleTesting


class BonferroniMultipleTesting(AbstractMultipleTesting):
    """
    Bonferroni correction method for multiple testing.
    """

    @classmethod
    def adjust(cls, p_values):
        """
        Adjust p-values using Bonferroni correction.

        :param p_values: list of raw p-values between 0 and 1.
        :return: list of adjusted p-values.
        """

        cls._validate_p_values(p_values)
        m = len(p_values)
        p_values_adjusted = [min(p_value * m, 1.0) for p_value in p_values]
        return p_values_adjusted

    @classmethod
    def test(cls, p_values, threshold=0.05):
        """
        Test hypotheses using Bonferroni correction.

        :param p_values: list of raw p-values for hypothesis testing.
        :param threshold: significance level for controlling FWER.
        :return: tuple containing boolean list of rejected hypotheses and list of adjusted p-values.
        """

        return super().test(p_values, threshold)


class SidakMultipleTesting(AbstractMultipleTesting):
    """
    Šidák correction method for multiple testing.
    """

    @classmethod
    def adjust(cls, p_values):
        """
        Adjust p-values using Šidák correction.

        :param p_values: list of raw p-values between 0 and 1.
        :return: list of adjusted p-values.
        :note: Requires independence or PLOD assumption between p-values.
        """

        cls._validate_p_values(p_values)
        m = len(p_values)
        p_values_adjusted = [1 - (1 - p_value) ** m for p_value in p_values]
        return p_values_adjusted

    @classmethod
    def test(cls, p_values, threshold=0.05):
        """
        Test hypotheses using Šidák correction.

        :param p_values: list of raw p-values for hypothesis testing.
        :param threshold: significance level for controlling FWER.
        :return: tuple containing boolean list of rejected hypotheses and list of adjusted p-values.
        :note: Requires independence or PLOD assumption between p-values.
        """

        return super().test(p_values, threshold)
