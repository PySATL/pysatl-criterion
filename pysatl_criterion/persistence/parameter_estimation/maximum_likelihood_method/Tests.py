import unittest
from MaximumLikelihoodMethod import likelihoodFunctionForNormalDistribution

class TestNormalDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = list(range(-108, 109))
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood > 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = [0.25, 0.26, 0.23]
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood > 0, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = list(range(10000))
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood > 0, "Likelihood should be positive for large data set")

    def test_zero_data(self):
        data = []
        with self.assertRaises(ZeroDivisionError):
            likelihoodFunctionForNormalDistribution(data)

    def test_all_values_equal(self):
        data = [5, 5, 5, 5, 5]
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood > 0, "Likelihood should be positive when all values equal")

class TestExponentialDistribution(unittest.TestCase):
    def test_big_distribution(self):
        r = 0

    def test_small_distribution(self):
        r = 0

    def test_a_lot_of_data(self):
        r = 0

    def test_zero_data(self):
        r = 0

    def test_all_values_equal(self):
        r = 0

if __name__ == '__main__':
    unittest.main()
