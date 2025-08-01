import unittest
from pysatl_criterion.persistence.parameter_estimation.maximum_likelihood_method.function_for_maximum_likelihood_method import *

#1.Uniform continuous distribution
class TestUniformContinuousDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForUniformContinuousDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForUniformContinuousDistribution(data)
        self.assertTrue(likelihood >= 10000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForUniformContinuousDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForUniformContinuousDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for large data set")

#2.NormalDistribution
class TestNormalDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForNormalDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

#3.Lognormal distribution
class TestLognormalDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForLognormalDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForLognormalDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForLognormalDistribution(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForLognormalDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForLognormalDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

# 4. Exponential distribution
class TestExponentialDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForExponentialDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = ([100, 200, 100])
        likelihood = likelihoodFunctionForExponentialDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForExponentialDistribution(data)
        self.assertTrue(likelihood >= 1, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForExponentialDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([0, 0, 0, 0, 0])
        likelihood = likelihoodFunctionForExponentialDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal 0")

# 5.Weibull distribution
class TestWeibullDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForWeibullDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForWeibullDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForWeibullDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood didn't find.")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForWeibullDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([1, 1, 1, 1, 1])
        likelihood = likelihoodFunctionForWeibullDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

#6.Gamma distribution
class TestGammaDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForGammaDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForGammaDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForGammaDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForGammaDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForGammaDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

#7.Beta distribution
class TestBetaDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForBetaDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForBetaDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForBetaDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForBetaDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForBetaDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

#8.Cauchy distribution
class TestCauchyDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForCauchyDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForCauchyDistribution(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForCauchyDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        likelihood = likelihoodFunctionForCauchyDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

#9.Chi-square distribution
class TestChiSquareDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForChiSquared(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForChiSquared(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForChiSquared(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForChiSquared(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForChiSquared(data)
        self.assertTrue(likelihood >= 100, "Likelihood should be positive when all values equal")

#10.Student distribution (t-distribution)
class TestTDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForTDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForTDistribution(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForTDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForTDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be positive when all values equal")

#11.Phisher distribution
class TestPhisherDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForFisherDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForFisherDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForFisherDistribution(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForFisherDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForFisherDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be positive when all values equal")

#12.Rayleigh distribution
class TestRayleighDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForRayleighDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForRayleighDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForRayleighDistribution(data)
        self.assertTrue(likelihood >= 10, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForRayleighDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForRayleighDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive when all values equal")

#13.Wigner distribution
class TestWignerDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForWignerDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForWignerDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForWignerDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_likelihood_is_zero_when_optimization_fails(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForWignerDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values equal")

#14.Pareto distribution
class TestParetoDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([0, -100, 100, -200])
        likelihood = likelihoodFunctionForParetoDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero for negative data")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForParetoDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForParetoDistribution(data)
        self.assertTrue(likelihood >= 10000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(1, 10000)))
        likelihood = likelihoodFunctionForParetoDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForParetoDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive when all values equal")

#15.Laplace distribution
class TestLaplaceDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForLaplaceDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForLaplaceDistribution(data)
        self.assertTrue(likelihood >= 1000, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForLaplaceDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForLaplaceDistribution(data)
        self.assertTrue(likelihood == np.inf, "Likelihood should be positive when all values equal")

#Discrete distributions:

#1.Discrete uniform distribution
class TestDiscreteUniformDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 10.5, -200])
        likelihood = likelihoodFunctionForDiscreteUniformDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForDiscreteUniformDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_a_lot_of_data(self):
        data = np.array(list(range(10000)))
        likelihood = likelihoodFunctionForDiscreteUniformDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for large data set")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForDiscreteUniformDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive when all values equal")

#2.Bernoulli distribution
class TestBernoulliDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForBernoulliDistribution(data)
        self.assertTrue(likelihood == 0, "Likelihood should be zero when all values are not 1 or 0")

    def test_small_distribution(self):
        data = np.array([0, 1, 0])
        likelihood = likelihoodFunctionForBernoulliDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_all_values_equal(self):
        data = np.array([1, 1, 1, 1, 1])
        likelihood = likelihoodFunctionForBernoulliDistribution(data)
        self.assertTrue(likelihood == 1, "Likelihood should be positive when all values equal")

#3.Binomial distribution
class TestBinomialDistribution(unittest.TestCase):
    def test_negative_data(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForBinomialDistribution(data, len(data))
        self.assertTrue(likelihood == 0, "Data values must be positive")

    def test_non_integer_values(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForBinomialDistribution(data, len(data))
        self.assertTrue(likelihood == 0, "Data values must be integer")

    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForBinomialDistribution(data, len(data))
        self.assertTrue(likelihood == 0, "Data values must be between 0 and n")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForBinomialDistribution(data, len(data))
        self.assertTrue(likelihood >= 0, "Likelihood should be positive when all values equal")

#4.Poisson distribution
class TestPoissonDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForPoissonDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([25, 26, 23])
        likelihood = likelihoodFunctionForPoissonDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_negative_data(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForPoissonDistribution(data)
        self.assertTrue(likelihood == 0, "Data values must be positive")

    def test_non_integer_values(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForPoissonDistribution(data)
        self.assertTrue(likelihood == 0, "Data values must be integer")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForPoissonDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive when all values equal")

#5.Geometric distribution
class TestGeometricDistribution(unittest.TestCase):
    def test_big_distribution(self):
        data = np.array([100, 200, 100])
        likelihood = likelihoodFunctionForGeometricDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for big distribution")

    def test_small_distribution(self):
        data = np.array([25, 26, 23])
        likelihood = likelihoodFunctionForGeometricDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive for small distribution")

    def test_negative_data(self):
        data = np.array([-100, 100, -200])
        likelihood = likelihoodFunctionForGeometricDistribution(data)
        self.assertTrue(likelihood == 0, "Data values must be positive")

    def test_non_integer_values(self):
        data = np.array([0.25, 0.26, 0.23])
        likelihood = likelihoodFunctionForGeometricDistribution(data)
        self.assertTrue(likelihood == 0, "Data values must be integer")

    def test_all_values_equal_to_mathExpect(self):
        data = np.array([5, 5, 5, 5, 5])
        likelihood = likelihoodFunctionForGeometricDistribution(data)
        self.assertTrue(likelihood >= 0, "Likelihood should be positive when all values equal")

if __name__ == '__main__':
    unittest.main()
