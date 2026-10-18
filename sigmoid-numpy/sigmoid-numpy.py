import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """

    exp_array = 1/np.exp(x)

    return 1/(1 + exp_array)