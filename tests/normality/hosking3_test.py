import pytest as pytest

from stattest.test.normal import Hosking3NormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.9515396,
                0.4302541,
                0.1149620,
                1.7218222,
                -0.4061157,
                -0.2528552,
                0.7840704,
                -1.6576825,
            ],
            41.33229,
        ),
        (
            [
                -1.4387336,
                1.2636724,
                -1.9232885,
                0.5963312,
                0.1208620,
                -1.1269378,
                0.5032659,
                0.3810323,
                0.8924223,
                1.8037073,
            ],
            117.5835,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 10.26769536963889),  # Negative values test
    ],
)
class TestCaseHosking3NormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return Hosking3NormalityTest()
