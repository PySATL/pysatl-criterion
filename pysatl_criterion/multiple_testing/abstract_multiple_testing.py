from abc import ABC, abstractmethod


class AbstractMultipleTesting(ABC):
    @staticmethod
    @abstractmethod
    def test(p_values: list[float], threshold: float) -> tuple[list[bool], list[float]]:
        """
        Perform multiple testing correction and return both rejection decisions
        and adjusted p-values.
        :param p_values: List of raw p-values for hypothesis testing
        :param threshold: Significance level for controlling FWER (Family-Wise Error Rate)
        or FDR (False Discovery Rate)
        :return: Tuple containing:
                - Boolean list indicating rejected hypotheses (True where rejected)
                - List of adjusted p-values after multiple testing correction
        """
        raise NotImplementedError("Method is not implemented")

    @staticmethod
    @abstractmethod
    def adjust(p_values: list[float]) -> list[float]:
        """
        Compute adjusted p-values for multiple testing correction.
        :param p_values: List of raw p-values for hypothesis testing
        :return: List of adjusted p-values after multiple testing correction
        """
        raise NotImplementedError("Method is not implemented")
