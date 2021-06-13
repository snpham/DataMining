import numpy as np


def manhattan_distance(x, y):
    return sum(np.abs(a-b) for a, b in zip(x, y))


def hamming_distance(s1, s2):
    """return the Hamming distance b/t equal-length sequences
    """
    if len(s1) != len(s2):
        raise ValueError("undefined for sequences of unequal length")
    
    result = sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    return (len(s1) - result) / len(s1)


def euclidean_distance(x, y):
    return np.sqrt(sum((a-b)**2 for a, b in zip(x ,y)))


def minkowski_distance(x, y, p_value):

    sum_val = sum(np.abs(a-b)**p_value for a, b in zip(x, y))
    root_val = 1 / p_value
    result = np.round(sum_val**root_val, 3)

    return result


def cosine_similarity(x,y):
    numerator = sum(a*b for a, b in zip(x,y))
    sqrtx = round(np.sqrt(sum([a*a for a in x])), 3)
    sqrty = round(np.sqrt(sum([a*a for a in y])), 3)
    denom = sqrtx*sqrty
    result = round(numerator/denom, 3)

    return result


if __name__ == '__main__':

    # manhattan distance
    result = manhattan_distance([10, 20, 10], [10, 20, 20])
    print(result)

    # hamming distance
    result = hamming_distance('CATCATCATCATCATCATCTTTTT',
                              'CATCATCTTCATCATCATCTTTTT')
    print(result)

    # hamming distance 2
    result = hamming_distance('ATGCATCATCATCATCATCTTTTT',
                              'CATCATCTTCATCATCATCTTTTT')
    print(result)

    # euclidean distance
    result = euclidean_distance([0, 3, 4, 5], [7, 6, 3, -1])
    print(result)

    # minkowski distance
    result = minkowski_distance([0, 3, 4, 5], [7, 6, 3, -1], 3)
    print(result)

    # minkowski distance
    result = cosine_similarity([5, 0, 3, 0, 2, 0, 0, 2, 0, 0], [3, 0, 2, 0, 1, 1, 0, 1, 0, 1])
    print(result)