from enum import Enum


class TestMethod(Enum):
    # Disables this test from being run
    __test__ = False

    """
    Test methods for hypotheses.
    """

    CRITICAL_VALUE = "critical_value"
    P_VALUE = "p_value"
