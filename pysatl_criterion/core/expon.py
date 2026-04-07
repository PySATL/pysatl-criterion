from scipy.stats import expon


def generate_expon(size, lam=1):  # TODO: refactor structure with inheritance??
    """
    Generate random variates from exponential distribution.

    :param size: number of samples to generate.
    :param lam: rate parameter (lambda) of exponential distribution.

    :return: array of generated random variates.
    """
    scale = 1 / lam
    return expon.rvs(size=size, scale=scale)


def cdf_expon(rvs, lam=1):
    """
    Calculate cumulative distribution function for exponential distribution.

    :param rvs: random variates or value to evaluate CDF at.
    :param lam: rate parameter (lambda) of exponential distribution.

    :return: CDF values for given random variates.
    """
    scale = 1 / lam
    return expon.cdf(rvs, scale=scale)
