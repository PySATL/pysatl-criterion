import pytest as pytest

from stattest.test.normal import LillieforsNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([-1, 0, 1], 0.17467808),
        (
            [
                0.8366388,
                1.1972029,
                0.4660834,
                -1.8013118,
                0.8941450,
                -0.2602227,
                0.8496448,
            ],
            0.2732099,
        ),
        (
            [
                0.72761915,
                -0.02049438,
                -0.13595651,
                -0.12371837,
                -0.11037662,
                0.46608165,
                1.25378956,
                -0.64296653,
                0.25356762,
                0.23345769,
            ],
            0.1695222,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.15073232084833066),  # Negative values test
    ],
)
class TestCaseLillieforsTestNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return LillieforsNormalityTest()
