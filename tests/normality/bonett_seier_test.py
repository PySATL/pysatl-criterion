import pytest as pytest

from stattest.test.normal import BonettSeierNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.7826974,
                -1.1840876,
                1.0317606,
                -0.7335811,
                -0.0862771,
                -1.1437829,
                0.4685360,
            ],
            -1.282991,
        ),
        (
            [
                0.73734236,
                0.40517722,
                0.09825027,
                0.27044629,
                0.93485784,
                -0.41404827,
                -0.01128772,
                0.41428093,
                0.18568170,
                -0.89367267,
            ],
            0.6644447,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], -0.2991304162412471),  # Negative values test
    ],
)
class TestCaseBonettSeierNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return BonettSeierNormalityTest()
