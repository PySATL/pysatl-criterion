import pytest as pytest

from stattest.test.normal import SWNormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        ([16, 18, 16], 0.75),
        ([16, 18, 16, 14], 0.9446643),
        ([16, 18, 16, 14, 15], 0.955627),
        ([38.7, 41.5, 43.8, 44.5, 45.5, 46.0, 47.7, 58.0], 0.872973),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 0.935564736597946),  # Negative values test
    ],
)
class TestCaseSWNormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return SWNormalityTest()
