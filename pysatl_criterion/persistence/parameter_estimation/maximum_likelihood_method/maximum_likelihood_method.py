import inspect

import function_for_maximum_likelihood_method
import numpy as np


def findMaxProbabilityFunction(data):
    """
    Find the distribution function with maximum likelihood for given data.

    :param data: list of observed data samples for distribution fitting.
    :return: likelihood function with the maximum value for the given data.
    :raises ValueError: if no function produces a positive likelihood value.
    """
    functions = [
        f for _, f in inspect.getmembers(function_for_maximum_likelihood_method, inspect.isfunction)
    ]

    max_func = None
    max_probability = 0

    for func in functions:
        result = func(data)
        if result > max_probability:
            max_probability = result
            max_func = func

    return max_func


def findResultOfLikelihoodFunction(data):
    """
    Validate data and find the best-fitting distribution function with its likelihood value.

    :param data: list of observed data samples for distribution fitting.
    :return: tuple containing the best likelihood function and its likelihood value.
    :raises ValueError: if data is empty or no function produces positive likelihood.
    :raises TypeError: if data contains non-numeric values.
    """
    data_array = np.array(data)

    if not data_array:
        raise ValueError("There are no data.")

    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError("Data should be numeric.")

    findMaxProbabilityFunction(data)
