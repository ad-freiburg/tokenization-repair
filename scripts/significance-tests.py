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

SCALING_FACTOR = 1
# SCALING_FACTOR = 2**10000


def precompute_binomials(max_n):
    """
    Precompute "n over k" times 2**-n for all n = 0, ..., max_n

    >>> binomials = precompute_binomials(5)
    >>> [binomials[5][k] * 32 for k in range(6)]
    [1.0, 5.0, 10.0, 10.0, 5.0, 1.0]
    """

    B = [[1]]
    for n in range(1, max_n + 1):
        B.append([0 for _ in range(n + 1)])
        for k in range(n + 1):
            if k == 0 or k == n:
                B[n][k] = B[n - 1][0] / 2
            else:
                B[n][k] = (B[n - 1][k - 1] + B[n - 1][k]) / 2
    return B


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
        if SCALING_FACTOR > 2**100:
            return binomial(n - 1, 0) // 2
        else:
            return binomial(n - 1, 0) / 2
    else:
        if SCALING_FACTOR > 2**100:
            return (binomial(n - 1, k - 1) + binomial(n - 1, k)) // 2
        else:
            return (binomial(n - 1, k - 1) + binomial(n - 1, k)) / 2


def accuracy_r_test_exact(A, B, binomials=precompute_binomials(5)):
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
            p_value += delta * binomials[n1][k1] * binomials[n2][k2]
            # p_value += delta * binomial(n1, k1) * binomial(n2, k2)
    return p_value 
    # return (p_value * 10**10 // (SCALING_FACTOR * SCALING_FACTOR)) / 10**10


def accuracy_r_test_sampling(acc1, acc2, num_samples=1024):
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
    if len(sys.argv) < 2 or len(sys.argv) > 6:
        print()
        print("Usage: python3 significance-tests.py <directory> "
              "[<parameter setting>] [<method 1>] [<method 2>] [<num samples>]")
        print()
        print("Read all files from the given directory, where each file "
              "has an odd number of lines in the the following format:")
        print()
        print("Line 1   : Parameter setting")
        print("Line 2k  : Method name")
        print("Line 2k+1: Accuracies (string of 0 or 1)")
        print()
        print("If only the directory is specified, then for each parameter "
              "setting, compare all pairs of methods and print those where "
              "the difference in average accuracy is significant")
        print()
        print("If a parameter setting is specified, compare all pairs of "
              "methods for that parameter setting")
        print()
        print("If in addition, one method is specified, compare that method "
              "to all other methods in that parameter setting; if two methods "
              "are specified, compare only those two methods")
        print()
        sys.exit(1)
    dir_name = sys.argv[1]
    compare_all_methods = (len(sys.argv) == 2)
    arg_parameter_setting = sys.argv[2] if len(sys.argv) >= 3 else None
    arg_method_name_1 = sys.argv[3] if len(sys.argv) >= 4 else None
    arg_method_name_2 = sys.argv[4] if len(sys.argv) >= 5 else None
    num_samples = int(sys.argv[5]) if len(sys.argv) >= 6 else 1024
    print("Number of samples used for randomization test: %d" % num_samples)
    print("Parameter setting: %s" % arg_parameter_setting)
    print("Method 1: %s" % arg_method_name_1)
    print("Method 2: %s" % arg_method_name_2)
    print("Compare all methods: ", compare_all_methods)
    sys.setrecursionlimit(10**6)

    # Precompute binomials
    max_n = 10000
    print("Precomputing binomial coefficients with max_n = %d ... " % max_n, end="")
    sys.stdout.flush()
    binomials = precompute_binomials(max_n)
    print("done")

    # Read all files in the given directory into an array of triples, where each
    # triple consists of the parameter string (first line of the file), the
    # method name (second line of the file), and the 0-1 list (third line of the
    # file).
    results = {}
    for file_name in os.listdir(dir_name):
        with open(os.path.join(dir_name, file_name)) as f:
            parameter_setting = f.readline().rstrip().replace(
                "expected token errors", "tokos").replace(
                "1.0 tokos", "100% tokos").replace(
                "0.1 tokos", "10% tokos")
            results[parameter_setting] = {}
            while True:
                method_name = f.readline().rstrip()
                if method_name == "":
                    break
                accuracies = list(map(int, f.readline().rstrip()))
                results[parameter_setting][method_name] = accuracies
    # print(methods_with_results)
    parameter_settings = list(results.keys())
    method_names = list(results[list(results.keys())[0]].keys())
    print()
    print("All parameter settings: ", parameter_settings)
    print("All method names: ", method_names)
    print()

    # Key of the parameter settings and method names used to sort them in the same
    # order as in our table
    ps_keys = { 'no typos, 10% tokos': 1, 'no typos, 100% tokos': 2, 'no typos, no spaces': 3,
                '10% typos, 10% tokos': 4, '10% typos, 100% tokos': 5, '10% typos, no spaces': 6 }
    mn_keys = { 'greedy': 1, 'bigrams': 2, 'bidirectional': 3, 'bidirectional robust': 4,
                'BS fw': 5, 'BS bw': 6, 'BS fw robust': 7, 'BS bw robust': 8,
                'two-pass': 9, 'two-pass robust': 10, 'BS fw+bi': 11, 'BS fw+bi robust': 12 }

    # Iterate over all or some combinations and show the difference in the
    # accuracy and the p-value for some or a selection, depending on the input
    # arguments (see usage info above).
    if arg_parameter_setting != None:
        parameter_settings = [arg_parameter_setting]
    method_names_1 = method_names if arg_method_name_1 == None else [arg_method_name_1]
    method_names_2 = method_names if arg_method_name_2 == None else [arg_method_name_2]
    for parameter_setting in sorted(parameter_settings, key = lambda x: ps_keys[x]):
        print("\x1b[1m%s\x1b[0m" % parameter_setting)
        print()
        for method_name_1 in sorted(method_names_1, key = lambda x: mn_keys[x]):
            for method_name_2 in sorted(method_names_2, key = lambda x: mn_keys[x]):
                if len(method_names_1) > 1 and len(method_name_2) > 1 and \
                        mn_keys[method_name_1] >= mn_keys[method_name_2]:
                    continue
                acc_1 = results[parameter_setting][method_name_1]
                acc_2 = results[parameter_setting][method_name_2]
                assert len(acc_1) == len(acc_2)
                mean_1 = sum(acc_1) / len(acc_1)
                mean_2 = sum(acc_2) / len(acc_2)
                if compare_all_methods:
                    p_value_exact = accuracy_r_test_exact(acc_1, acc_2, binomials)
                    if p_value_exact > 0.01:
                        print("%24s ... %21s: %5.2f%%" %
                                (method_name_1, parameter_setting, 100 * mean_1))
                        print("%24s ... %21s: %5.2f%%" %
                                (method_name_2, parameter_setting, 100 * mean_2))
                        print("%50s: %5.2f%%" %
                                ("difference", 100 * abs(mean_1 - mean_2)))
                        print("%50s:  %.3f" % ("p-value exact", p_value_exact))
                        print()
                else:
                    print("%24s ... %21s: %5.2f%%" %
                            (method_name_1, parameter_setting, 100 * mean_1))
                    print("%24s ... %21s: %5.2f%%" %
                            (method_name_2, parameter_setting, 100 * mean_2))
                    print("%50s: %5.2f%%" %
                            ("difference", 100 * abs(mean_1 - mean_2)))
                    p_value_sampled = accuracy_r_test_sampling(acc_1, acc_2, num_samples)
                    print("%50s:  %.3f" % ("p-value sampled", p_value_sampled))
                    p_value_exact = accuracy_r_test_exact(acc_1, acc_2, binomials)
                    print("%50s:  %.3f" % ("p-value exact", p_value_exact))
                    print()

