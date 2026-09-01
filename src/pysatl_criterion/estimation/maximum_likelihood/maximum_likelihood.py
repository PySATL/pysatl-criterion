import inspect

import numpy as np

from . import function_for_maximum_likelihood


def _single_sample_likelihood_functions():
    """
    Collect the likelihood functions that can be evaluated from a data sample alone.

    Only functions defined in :mod:`function_for_maximum_likelihood` whose name starts
    with ``likelihood_function_`` and that take a single required positional argument
    are eligible. This deliberately excludes helpers imported into the module
    (``minimize_scalar``, ``root_scalar``, ...) and functions that need extra
    parameters such as ``likelihood_function_binomial(data, n)``.
    """
    candidates = []
    module_name = function_for_maximum_likelihood.__name__
    for name, func in inspect.getmembers(function_for_maximum_likelihood, inspect.isfunction):
        if not name.startswith("likelihood_function_"):
            continue
        if getattr(func, "__module__", None) != module_name:
            continue
        required = [
            parameter
            for parameter in inspect.signature(func).parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ]
        if len(required) == 1:
            candidates.append(func)
    return candidates


def find_max_probability_function(data):
    """
    Find the distribution function with maximum likelihood for given data.

    :param data: list of observed data samples for distribution fitting.
    :return: likelihood function with the maximum value for the given data,
        or ``None`` if no function produces a positive likelihood value.
    """
    data = np.asarray(data)
    max_func = None
    max_probability = 0

    for func in _single_sample_likelihood_functions():
        result = func(data)
        if result > max_probability:
            max_probability = result
            max_func = func

    return max_func


def find_result_of_likelihood_function(data):
    """
    Validate data and find the best-fitting distribution function with its likelihood value.

    :param data: list of observed data samples for distribution fitting.
    :return: tuple containing the best likelihood function and its likelihood value.
    :raises ValueError: if data is empty or no function produces positive likelihood.
    :raises TypeError: if data contains non-numeric values.
    """
    data_array = np.array(data)

    if data_array.size == 0:
        raise ValueError("There are no data.")

    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError("Data should be numeric.")

    likelihood_function = find_max_probability_function(data_array)
    if likelihood_function is None:
        raise ValueError("No suitable likelihood function was found.")

    return likelihood_function, likelihood_function(data_array)
