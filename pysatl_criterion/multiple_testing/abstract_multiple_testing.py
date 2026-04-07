from abc import ABC, abstractmethod


class AbstractMultipleTesting(ABC):
    @staticmethod
    def _validate_p_values(p_values: list[float]) -> None:
        """
        Validate that all p-values are in [0,1] range.
        :param p_values: List of p-values for hypothesis testing
        """
        if not all(0 <= x <= 1 for x in p_values):
            raise ValueError("All p-values must be in range [0,1].")

    @classmethod
    def test(cls, p_values: list[float], threshold: float = 0.05) -> tuple[list[bool], list[float]]:
        """
        Perform multiple testing correction and return both rejection decisions
        and adjusted p-values.
        :param p_values: List of raw p-values for hypothesis testing
        :param threshold: Significance level for controlling FWER (Family-Wise Error Rate)
        or FDR (False Discovery Rate) (default is 0.05)
        :return: Tuple containing:
                - Boolean list indicating rejected hypotheses (True where rejected)
                - List of adjusted p-values after multiple testing correction
        """
        cls._validate_p_values(p_values)
        p_values_adjusted = cls.adjust(p_values)
        rejected = [p_value < threshold for p_value in p_values_adjusted]
        return rejected, p_values_adjusted

    @classmethod
    @abstractmethod
    def adjust(cls, p_values: list[float]) -> list[float]:
        """
        Compute adjusted p-values for multiple testing correction.
        :param p_values: List of raw p-values for hypothesis testing
        :return: List of adjusted p-values after multiple testing correction
        """
        raise NotImplementedError("Method is not implemented")
