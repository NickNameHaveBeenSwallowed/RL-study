import numpy as np

def value_reward_transfomer(x):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1)

def inverse_value_reward_transfomer(x):
    return 2 * x + np.sign(x) * x ** 2

def discrete_transform(x, discrete_support_size, floor, ceiling):
    transform_x = np.zeros((discrete_support_size))
    if x <= floor:
        transform_x[0] = 1
        return transform_x
    elif x >= ceiling:
        transform_x[-1] = 1
        return transform_x
    else:
        low = np.floor(x).astype(int) + ceiling
        high = np.floor(x).astype(int) + ceiling + 1
        difference = x - np.floor(x)

        transform_x[low] = 1 - difference
        transform_x[high] = difference
        return transform_x

def inverse_discrete_transform(transform_x, discrete_support_size, ceiling):
    return np.dot(transform_x, range(discrete_support_size)) - ceiling