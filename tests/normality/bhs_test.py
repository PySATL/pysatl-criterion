import pytest as pytest

from stattest.test.normal import BHSNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -1.3662750,
                -1.2772617,
                1.2341902,
                0.4943849,
                -0.6015152,
                0.7927679,
                -2.6990387,
            ],
            0.9122889,
        ),
        (
            [
                2.39179539,
                1.00572055,
                0.55561602,
                0.49246060,
                -0.43379600,
                0.03081284,
                0.31172966,
                0.40097292,
                0.46238934,
                -0.29856372,
            ],
            1.239515,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.16232061118184815),  # Negative values test
    ],
)
# TODO: remove skip
@pytest.mark.skip(reason="no way of currently testing this")
class TestCaseBHSNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return BHSNormalityTest()
