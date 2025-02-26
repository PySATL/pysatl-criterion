import pytest as pytest

from stattest.test.normal import SpiegelhalterNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.31669223,
                2.22819380,
                -0.27391944,
                -1.57616900,
                -2.21675399,
                -0.01497801,
                -1.65492071,
            ],
            1.328315,
        ),
        (
            [
                -1.6412500,
                -1.1946111,
                1.1054937,
                -0.4210709,
                -1.1736754,
                -1.1750840,
                1.3267088,
                -0.3299987,
                -0.5767829,
                -1.4114579,
            ],
            1.374628,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 1.3125438497485047),  # Negative values test
    ],
)
class TestCaseSpiegelhalterNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return SpiegelhalterNormalityTest()
