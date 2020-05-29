"""
Tokenization Repair Paper, Submission to EMNLP 2020 

Script for testing the statistical significance of our results.
"""

import sys
import re
import math
import random
import os
import functools

SCALING_FACTOR = 2**10000


@functools.lru_cache(None)
def binomial(n, k):
    """
    Compute the binomial coefficient "n over k" times 2**-n recursively, using
    functools.lru_cache to cache result. That way, we don't need an explicit
    precomputation and we don't need to specify a bound on n.
    
    >>> [binomial(5, k) * 32 for k in range(6)]
    [1.0, 5.0, 10.0, 10.0, 5.0, 1.0]
    >>> sys.setrecursionlimit(10**6)
    >>> binomial(1075, 0)
    0.0
    """

    assert n >= 0 and (k >= 0 and k <= n)
    if n == 0:
        return 1 * SCALING_FACTOR
    if k == 0 or k == n:
        return binomial(n - 1, 0) // 2
    else:
        return (binomial(n - 1, k - 1) + binomial(n - 1, k)) // 2


def accuracy_r_test_exact(A, B):
    """
    Given two 0-1 sequences of the same length n (measuring accuracy, in our
    case sequence accuracies), compute the p-value using a paired R test.

    A paired R test considers all 2^n combinations of mapping each of the n
    measurement pairs to A and B (for a pair of measurements x, y we can either
    assign x to A and y to B or vice versa).

    >>> accuracy_r_test_exact([0], [1])
    1.0
    >>> accuracy_r_test_exact([0, 0, 0], [1, 0, 0])
    1.0
    >>> accuracy_r_test_exact([0, 0], [1, 1])
    0.5
    """
    assert len(A) == len(B)
    n = len(A)
    n1 = sum(A[i] > B[i] for i in range(n))  # Number of pairs (1, 0)
    n2 = sum(B[i] > A[i] for i in range(n))  # Number of pairs (0, 1)
    p_value = 0
    for k1 in range(n1 + 1):
        for k2 in range(n2 + 1):
            # Contribution of assignment, where k1 of the (1, 0) pairs and k2 of
            # the (0, 1) pairs are flipped.
            delta = abs((n1 - n2) - 2 * (k1 - k2)) >= abs(n1 - n2)
            # Multiply with the number of such assignements, which is (n over
            # k1) times (n over k2), times 2 ** (n - n1 - n2) because swapping
            # any of the n - n1 - n2 other pairs does not matter, divided by the
            # total number of assignments 2 ** n.
            p_value += delta * binomial(n1, k1) * binomial(n2, k2)
    return (p_value // (SCALING_FACTOR * SCALING_FACTOR // 10**10)) / 10**10


def accuracy_r_test_sampling(acc1, acc2, num_samples=128):
    """
    Like above, but explicitly iterating over all assignments or using random
    sampling if there are too many.

    >>> accuracy_r_test_sampling([0], [1])
    1.0
    >>> accuracy_r_test_sampling([0, 0, 0], [1, 0, 0])
    1.0
    >>> accuracy_r_test_sampling([0, 0], [1, 1])
    0.5
    """

    assert len(acc1) == len(acc2)
    n = len(acc1)
    use_random_sampling = (math.log2(num_samples) < n)
    if math.log2(num_samples) > n:
        num_samples = 2**n
    diff_observed = sum([acc1[i] - acc2[i] for i in range(n)]) / n
    # print("%40s: %5.2f%%" % ("difference", abs(100 * diff_observed)))

    # Now count the number of assignments for which the diff is >= the observed
    # diff.
    # 
    # NOTE: We consider absolute values, that is, a diff of -5.3 is considered
    # >= a diff of 4.2 although the diff is in the opposite direction. This
    # makes sense under the null hypothesis that the two sequences come from the
    # same random process.
    count = 0
    for i in range(num_samples):
        # Compute the next assignment as a 0-1 sequence of length n (where 1
        # means swapping the two respective elements from acc1 and acc2).
        if use_random_sampling:
            assignment = "".join([str(random.randint(0, 1)) for _ in range(n)])
        else:
            assignment = format(i, "0" + str(n) + "b")
        # Compute the mean difference between A and B using this assignment. In
        # the swap array, simply map "0" -> +1 and "1" -> - 1.
        swap = [1 if x == "0" else -1 for x in assignment]
        diff = sum([(acc1[i] - acc2[i]) * swap[i] for i in range(n)]) / n
        # print(diff, diff_observed)
        # print(swap, diff, diff_observed)
        if abs(diff) >= abs(diff_observed):
            count += 1

    return count / num_samples


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 significance-tests.py <directory> <num samples>"
              "<parameter string> <method 1> <method 2>")
        sys.exit(1)
    dir_name = sys.argv[1]
    num_samples = 128 if len(sys.argv) == 2 else int(sys.argv[2])
    parameter_string = sys.argv[3]
    method_1_name_given = sys.argv[4]
    method_2_name_given = sys.argv[5]
    print("Number of samples used for randomization test: %d" % num_samples)
    print("Parameter string: %s" % parameter_string)
    print("Method 1: %s" % method_1_name_given)
    print("Method 2: %s" % method_2_name_given)
    sys.setrecursionlimit(10**6)

    # Read all files in the given directory into an array of triples, where each
    # triple consists of the parameter string (first line of the file), the
    # method name (second line of the file), and the 0-1 list (third line of the
    # file).
    methods_with_results = []
    for file_name in os.listdir(dir_name):
        with open(os.path.join(dir_name, file_name)) as f:
            parameter_string_f = f.readline().rstrip().replace(
                "expected token errors", "tokos")
            if parameter_string_f != parameter_string:
                continue
            while True:
                method_name = f.readline().rstrip()
                if method_name == "":
                    break
                accuracies = list(map(int, f.readline().rstrip()))
                methods_with_results.append(
                        (parameter_string, method_name, accuracies))
    # print(methods_with_results)
    print("Methods: ", [x[1] for x in methods_with_results])
    print()

    # Now iterate over all pairs of methods, re-compute the means and compute
    # the p-value of the difference according to the paired two-sided R-Test.
    k = len(methods_with_results)
    for i in range(k):
        for j in range(k):
            parameters_1 = methods_with_results[i][0]
            parameters_2 = methods_with_results[j][0]
            method_name_1 = methods_with_results[i][1]
            method_name_2 = methods_with_results[j][1]
            if method_name_1 != method_1_name_given or method_name_2 != method_2_name_given:
                continue
            acc_1 = methods_with_results[i][2]
            acc_2 = methods_with_results[j][2]
            assert len(acc_1) == len(acc_2)
            mean_1 = sum(acc_1) / len(acc_1)
            mean_2 = sum(acc_2) / len(acc_2)
            print("%15s ... %20s: %5.2f%%" % (method_name_1, parameters_1, 100 * mean_1))
            print("%15s ... %20s: %5.2f%%" % (method_name_2, parameters_2, 100 * mean_2))
            p_value_sampled = accuracy_r_test_sampling(acc_1, acc_2, num_samples)
            p_value_exact = accuracy_r_test_exact(acc_1, acc_2)
            print("%40s:  %.3f" % ("p-value sampled", p_value_sampled))
            print("%40s:  %.3f" % ("p-value exact", p_value_exact))
            print()

