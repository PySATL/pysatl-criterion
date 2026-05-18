from enum import Enum


class TestMethod(Enum):
    """
    Test methods for hypotheses.
    """

    # Disables this test from being run
    __test__ = False

    CRITICAL_VALUE = "critical_value"
    P_VALUE = "p_value"
