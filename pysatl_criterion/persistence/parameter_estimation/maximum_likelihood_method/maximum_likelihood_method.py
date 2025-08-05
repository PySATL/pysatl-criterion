import inspect

import numpy as np
import function_for_maximum_likelihood_method

def findMaxProbabilityFunction(data):
    functions = [
        f for _, f in inspect.getmembers(
            function_for_maximum_likelihood_method,
            inspect.isfunction
        )
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
    data_array = np.array(data)

    if not data_array:
        raise ValueError("There are no data.")

    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError("Data should be numeric.")

    findMaxProbabilityFunction(data)
