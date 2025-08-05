from math import comb, exp, factorial, log, pi, sqrt

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from scipy.special import gammaln
from scipy.stats import beta, cauchy, chi2, f, gamma, rayleigh, t


# Continuous distributions:

#1.Uniform continuous distribution
#2.Normal distribution
#3.Lognormal distribution
#4.Exponential distribution
#5.Weibull distribution
#6.Gamma distribution
#7.Beta distribution
#8.Cauchy distribution
#9.Chi-square distribution
#10.Student distribution (t-distribution)
#11.phisher distribution
#12.Rayleigh distribution
#13.Wigner distribution
#14.Pareto distribution
#15.Laplace distribution

#Discrete distributions:

#1.Discrete uniform distribution
#2.Bernoulli distribution
#3.Binomial distribution
#4.Poisson distribution
#5.Geometric distribution

# Continuous distributions:

#1.Uniform continuous distribution
def likelihoodFunctionForUniformContinuousDistribution(dataArray):
    a = np.min(dataArray)
    b = np.max(dataArray)

    if a == b:
        return 0

    log_likelihood = -len(dataArray) * log(b - a)
    likelihood = exp(log_likelihood)

    return likelihood

# 2.Normal distribution
def likelihoodFunctionForNormalDistribution(dataArray):
    n = len(dataArray)
    mean = sum(dataArray) / n

    variance = sum((x - mean) ** 2 for x in dataArray) / n
    if variance == 0:
        return 0

    log_likelihood = 0
    for x in dataArray:
        prob_density = (1 / sqrt(2 * pi * variance)) * exp(- ((x - mean) ** 2) / (2 * variance))
        log_likelihood += log(prob_density)

    likelihood = exp(log_likelihood)

    return likelihood

#3.Lognormal distribution
def likelihoodFunctionForLognormalDistribution(dataArray):
    n = len(dataArray)
    if any(x <= 0 for x in dataArray):
        return 0

    log_data = [log(x) for x in dataArray]
    mean_log = sum(log_data) / n
    variance_log = sum((x - mean_log) ** 2 for x in log_data) / n
    if variance_log == 0:
        return 0

    log_likelihood = 0
    for x in dataArray:
        prob_density = (
                (1 / (x * sqrt(2 * pi * variance_log)))
                * exp(-((log(x) - mean_log) ** 2) / (2 * variance_log))
        )
        log_likelihood += log(prob_density)

    likelihood = exp(log_likelihood)

    return likelihood

# 4. Exponential distribution
def likelihoodFunctionForExponentialDistribution(dataArray):
    if any(x < 0 for x in dataArray):
        return 0

    if np.all(dataArray == 0):
        return 0

    n = len(dataArray)
    lam =  n / sum(dataArray)

    log_likelihood = 0
    for x in dataArray:
        prob_density = lam * exp(-lam * x)
        log_likelihood += log(prob_density)

    likelihood = exp(log_likelihood)

    return likelihood

# 5.Weibull distribution
def likelihoodFunctionForWeibullDistribution(dataArray):
    if np.any(dataArray <= 0):
       return 0

    A = np.mean(np.log(dataArray))

    def g(k):
        xk = dataArray ** k
        C = np.log(np.mean(xk))
        B = np.sum(xk * np.log(dataArray)) / np.sum(xk)
        return (1 - C) / k + A - B

    if g(0.1) * g(10) > 0:
        return 0

    res = root_scalar(g, bracket=[0.1, 10], method='bisect')
    k = res.root

    lam = (np.mean(dataArray ** k)) ** (1 / k)
    likelihood = np.prod((k / lam) * (dataArray / lam)**(k - 1) * np.exp(-(dataArray / lam)**k))

    return likelihood

#6.Gamma distribution
def likelihoodFunctionForGammaDistribution(dataArray):
    if np.any(dataArray <= 0) or np.all(dataArray == dataArray[0]):
        return 0

    shape, loc, scale = gamma.fit(dataArray, floc=0)

    log_likelihood = np.sum(gamma.logpdf(dataArray, a=shape, loc=loc, scale=scale))

    likelihood = np.exp(log_likelihood)

    return likelihood

#7.Beta distribution
def likelihoodFunctionForBetaDistribution(dataArray):
    if (len(dataArray) == 0
            or np.any((dataArray <= 0) | (dataArray >= 1))
            or np.all(dataArray == dataArray[0])):
        return 0

    alpha, beta_param, loc, scale = beta.fit(dataArray, floc=0, fscale=1)

    n = len(dataArray)

    log_likelihood = (
            n * (gammaln(alpha + beta_param) - gammaln(alpha) - gammaln(beta_param)) +
            (alpha - 1) * np.sum(np.log(dataArray)) +
            (beta_param - 1) * np.sum(np.log(1 - dataArray))
    )

    likelihood = exp(log_likelihood)

    return likelihood

#8.Cauchy distribution
def likelihoodFunctionForCauchyDistribution(dataArray):
    if np.all(dataArray == dataArray[0]):
        return 0

    loc, scale = cauchy.fit(dataArray)

    if scale <= 0:
        return 0

    log_likelihood = np.sum(cauchy.logpdf(dataArray, loc=loc, scale=scale))

    try:
        likelihood = np.exp(log_likelihood)
    except OverflowError:
        likelihood = float('inf')

    return likelihood

#9.Chi-square distribution
def likelihoodFunctionForChiSquared(dataArray):
    if np.any(np.array(dataArray) <= 0):
        return 0

    df, loc, scale = chi2.fit(dataArray, floc=0)

    log_likelihood = np.sum(chi2.logpdf(dataArray, df, loc=loc, scale=scale))

    likelihood = np.exp(log_likelihood)

    return likelihood

#10.Student distribution (t-distribution)
def likelihoodFunctionForTDistribution(dataArray):
    if np.all(dataArray == dataArray[0]):
        return 0

    df, loc, scale = t.fit(dataArray)
    log_likelihood = np.sum(t.logpdf(dataArray, df, loc=loc, scale=scale))

    return np.exp(log_likelihood)

#11.Phisher distribution
def likelihoodFunctionForFisherDistribution(dataArray):
    if np.any(dataArray <= 0) or np.all(dataArray == dataArray[0]):
        return 0

    dfn, dfd, loc, scale = f.fit(dataArray, floc=0)
    log_likelihood = np.sum(f.logpdf(dataArray, dfn, dfd, loc=loc, scale=scale))

    return np.exp(log_likelihood)

#12.Rayleigh distribution
def likelihoodFunctionForRayleighDistribution(dataArray):
    if np.any(dataArray < 0):
        return 0

    params = rayleigh.fit(dataArray, floc=0)
    scale = params[1]

    log_likelihood = np.sum(rayleigh.logpdf(dataArray, loc=0, scale=scale))
    return np.exp(log_likelihood)

#13.Wigner distribution
def likelihoodFunctionForWignerDistribution(dataArray):
    if np.all(dataArray == dataArray[0]):
        return 0

    max_abs_x = np.max(np.abs(dataArray))

    def neg_log_likelihood(R):
        if R <= max_abs_x:
            return np.inf
        n = len(dataArray)
        return -(n * log(2 / (pi * R**2)) + 0.5 * np.sum(np.log(R**2 - dataArray**2)))

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(max_abs_x + 1e-6, max_abs_x + 10),
        method='bounded'
    )

    if not result.success:
        return 0

    return exp(-result.fun)

#14.Pareto distribution
def likelihoodFunctionForParetoDistribution(dataArray):
    if np.any(dataArray <= 0):
        return 0

    n = len(dataArray)
    x_m = np.min(dataArray)

    log_ratios = np.log(dataArray / x_m)
    sum_log_ratios = np.sum(log_ratios)

    if sum_log_ratios == 0:
        return np.inf

    alpha = n / sum_log_ratios

    log_likelihood = (
            n * np.log(alpha) +
            n * alpha * np.log(x_m) -
            (alpha + 1) * np.sum(np.log(dataArray))
    )

    return exp(log_likelihood)

#15.Laplace distribution
def likelihoodFunctionForLaplaceDistribution(dataArray):
    n = len(dataArray)
    mu = np.median(dataArray)
    b = np.mean(np.abs(dataArray - mu))
    if b == 0:
        return np.inf

    log_likelihood = -n * np.log(2 * b) - (1 / b) * np.sum(np.abs(dataArray - mu))
    likelihood = np.exp(log_likelihood)

    return likelihood

#Discrete distributions:

#1.Discrete uniform distribution
def likelihoodFunctionForDiscreteUniformDistribution(dataArray):
    a = np.min(dataArray)
    b = np.max(dataArray)

    support_size = b - a + 1

    log_likelihood = -len(dataArray) * np.log(support_size)
    likelihood = np.exp(log_likelihood)

    return likelihood

#2.Bernoulli distribution
def likelihoodFunctionForBernoulliDistribution(dataArray):
    if not np.all(np.isin(dataArray, [0, 1])):
        return 0

    n = len(dataArray)
    k = np.sum(dataArray)
    p = k / n

    if p == 0 or p == 1:
        return 1

    log_likelihood = k * np.log(p) + (n - k) * np.log(1 - p)
    likelihood = np.exp(log_likelihood)

    return likelihood

#3.Binomial distribution
def likelihoodFunctionForBinomialDistribution(dataArray, n):
    if np.any((dataArray < 0) | (dataArray > n) | (~np.equal(np.mod(dataArray, 1), 0))):
        return 0

    m = len(dataArray)

    total_successes = np.sum(dataArray)
    p = total_successes / (m * n)

    if p == 0 or p == 1:
        if np.all(dataArray == 0) or np.all(dataArray == n):
            return 1
        else:
            return 0

    log_likelihood = np.sum([
        log(comb(n, x)) + x * log(p) + (n - x) * log(1 - p)
        for x in dataArray
    ])

    return exp(log_likelihood)

#4.Poisson distribution
def likelihoodFunctionForPoissonDistribution(dataArray):
    if np.any(dataArray < 0) or not np.all(np.floor(dataArray) == dataArray):
        return 0

    sum_x = np.sum(dataArray)
    lam = sum_x / len(dataArray)

    log_likelihood = sum(x * log(lam) - lam - log(factorial(x)) for x in dataArray)
    likelihood = exp(log_likelihood)

    return likelihood

#5.Geometric distribution
def likelihoodFunctionForGeometricDistribution(dataArray):
    if np.any(dataArray < 1) or not np.all(np.floor(dataArray) == dataArray):
        return 0

    n = len(dataArray)
    sum_x = np.sum(dataArray)
    p = n / sum_x

    log_likelihood = n * log(p) + (sum_x - n) * log(1 - p)
    likelihood = exp(log_likelihood)

    return likelihood
