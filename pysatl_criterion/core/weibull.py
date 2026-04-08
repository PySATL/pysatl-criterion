from scipy.stats import exponweib


def generate_weibull(size, a=1, k=5):
    """
    Generate random variates from Weibull distribution.

    :param size: number of samples to generate.
    :param a: shape parameter of Weibull distribution.
    :param k: shape parameter of Weibull distribution.

    :return: array of generated random variates.
    """
    return exponweib.rvs(a=a, c=k, size=size)


def generate_weibull_cdf(rvs, a=1, k=5):
    """
    Calculate cumulative distribution function for Weibull distribution.

    :param rvs: random variates or value to evaluate CDF at.
    :param a: shape parameter of Weibull distribution.
    :param k: shape parameter of Weibull distribution.

    :return: CDF values for given random variates.
    """
    return exponweib.cdf(rvs, a=a, c=k)


def generate_weibull_logcdf(rvs, a=1, k=5):
    """
    Calculate log of cumulative distribution function for Weibull distribution.

    :param rvs: random variates or value to evaluate log-CDF at.
    :param a: shape parameter of Weibull distribution.
    :param k: shape parameter of Weibull distribution.

    :return: log-CDF values for given random variates.
    """
    return exponweib.logcdf(rvs, a=a, c=k)


def generate_weibull_logsf(rvs, a=1, k=5):
    """
    Calculate log of survival function for Weibull distribution.

    :param rvs: random variates or value to evaluate log-SF at.
    :param a: shape parameter of Weibull distribution.
    :param k: shape parameter of Weibull distribution.

    :return: log-survival function values for given random variates.
    """
    return exponweib.logsf(rvs, a=a, c=k)
