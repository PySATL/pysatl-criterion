from abc import ABC, abstractmethod
from typing import List, Tuple


class AbstractMultipleTesting(ABC):
    @staticmethod
    @abstractmethod
    def test(p_values: List[float], threshold: float) -> Tuple[List[bool], List[float]]:
        """
        Perform multiple testing correction and return both rejection decisions and adjusted p-values.
        :param p_values: List of raw p-values for hypothesis testing
        :param threshold: Significance level for controlling FWER (Family-Wise Error Rate) or FDR 
        (False Discovery Rate)
        :return: Tuple containing:
                - Boolean list indicating rejected hypotheses (True where rejected)
                - List of adjusted p-values after multiple testing correction
        """
        raise NotImplementedError("Method is not implemented")

    @staticmethod
    @abstractmethod
    def adjust(p_values: List[float]) -> List[float]:
        """
        Compute adjusted p-values for multiple testing correction.
        :param p_values: List of raw p-values for hypothesis testing
        :return: List of adjusted p-values after multiple testing correction
        """
        raise NotImplementedError("Method is not implemented")
