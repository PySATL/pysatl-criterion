from math import sqrt, exp, pi, log

dataArray = [4, 3, 5, 8, 7, 5]

# Continuous distributions:
#1.Uniform continuous distribution
#2.Normal distribution
#3.Lognormal distribution
#4.Exponential distribution
#5.Weibull distribution
#6.Lognormal distribution
#7.Gamma distribution
#8.Beta distribution
#9.Cauchy distribution
#10.Chi-square distribution
#11.Student distribution (t-distribution)
#12.phisher distribution
#13.Rayleigh distribution
#14.Wigner distribution
#15.Pareto distribution
#16.Laplace distribution

#Discrete distributions:
#1.Discrete uniform distribution
#2.Bernoulli distribution
#3.Binomial distribution
#4.Poisson distribution
#5.Geometric distribution

# Continuous distributions:
# 2.Normal distribution
def likelihoodFunctionForNormalDistribution(data):
    n = len(data)
    mean = sum(data) / n

    variance = sum((x - mean) ** 2 for x in data) / n
    if variance == 0:
        variance = 1e-10

    log_likelihood = 0
    for x in data:
        prob_density = (1 / sqrt(2 * pi * variance)) * exp(- ((x - mean) ** 2) / (2 * variance))
        log_likelihood += log(prob_density)

    likelihood = exp(log_likelihood)
    return likelihood

# 4. Exponential distribution
def likelihoodFunctionForExponentialDistribution(data):
    n = len(data)
    l =  n / sum(data)

    log_likelihood = 0
    for x in data:
        prob_density = l * exp(-l * x)
        log_likelihood += log(prob_density)

    likelihood = exp(log_likelihood)
    return likelihood

# 5.Weibull distribution
# ...
