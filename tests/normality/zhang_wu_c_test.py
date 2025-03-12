import pytest as pytest
from numpy import nan

from stattest.test.normal import ZhangWuCNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.1750217,
                -1.8522318,
                -0.3997543,
                -0.3326987,
                -0.9864067,
                -0.4701900,
                2.7674965,
            ],
            5.194459,
        ),
        (
            [
                -0.12344697,
                -0.74936974,
                1.12023439,
                1.09091550,
                -0.05204564,
                -0.35421236,
                -0.70361281,
                2.38810563,
                -0.70401541,
                1.16743393,
            ],
            5.607312,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], nan),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 3.2388433290344802),  # Negative values test
    ],
)
class TestCaseZhangWuCNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return ZhangWuCNormalityTest()
