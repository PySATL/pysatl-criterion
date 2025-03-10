import pytest as pytest

from stattest.test.normal import Hosking4NormalityTest
from tests.normality.abstract_normality_test_case import AbstractNormalityTestCase


@pytest.mark.parametrize(
    ("data", "result"),
    [
        (
            [
                -0.93804525,
                -0.85812989,
                -1.35114261,
                0.16821566,
                2.05324842,
                0.72370964,
                1.58014787,
                0.07116436,
                -0.20992477,
                0.37184699,
                -0.41287789,
            ],
            1.737481,
        ),
        (
            [
                -0.18356827,
                0.42145728,
                -1.30305510,
                1.65498056,
                0.16475340,
                0.68201228,
                -0.26179821,
                -0.03263223,
                1.57505463,
                -0.34043549,
            ],
            3.111041,
        ),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # Zero exception test
        ([-4, -1, -6, -8, -4, -2, 0, -2, 0, -3], 100.68880246109997),  # Negative values test
    ],
)
class TestCaseHosking4NormalityTest(AbstractNormalityTestCase):
    @pytest.fixture
    def statistic_test(self):
        return Hosking4NormalityTest()
