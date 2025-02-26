import pytest as pytest

from stattest.test.normal import CoinNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                0.5171378,
                1.1163130,
                -1.3117699,
                -0.1739053,
                -0.6385798,
                -0.2520054,
                0.3237990,
            ],
            0.01249513,
        ),
        (
            [
                0.92728608,
                -0.75756591,
                -0.07266914,
                0.09636470,
                -1.13792085,
                -0.91534895,
                1.57469227,
                0.28462605,
                0.22804695,
                -0.29829152,
            ],
            0.0009059856,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.0005867649270227973),  # Negative values test
    ],
)
class TestCaseCoinNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return CoinNormalityTest()
