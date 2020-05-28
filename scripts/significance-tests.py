"""
Tokenization Repair Paper, Submission to EMNLP 2020 

Script for testing the statistical significance of our results.
"""

import sys
import re
import math
import random

def accuracy_r_test(acc1, acc2, num_samples=1024):
    """
    Given two 0-1 sequences of the same length n (measuring accuracy, in our
    case sequence accuracies), compute the p-value using a paired R test.

    A paired R test considers all 2^n combinations of mapping each of the n
    measurement pairs to A and B (for a pair of measurements x, y we can either
    assign x to A and y to B or vice versa).

    For 2^n > num_samples, num_samples combinations are picked at random.

    >>> accuracy_r_test([0], [1])
    1.0
    >>> accuracy_r_test([0, 0, 0], [1, 0, 0])
    1.0
    >>> accuracy_r_test([0, 0], [1, 1])
    0.5
    """

    assert len(acc1) == len(acc2)
    n = len(acc1)
    random_sampling = (math.log2(num_samples) < n)
    if math.log2(num_samples) > n:
        num_samples = 2**n
    diff_observed = sum([acc1[i] - acc2[i] for i in range(n)]) / n

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
        if random_sampling:
            assignment = random.getrandbits(n)
        else:
            assignment = format(i, "0" + str(n) + "b")
        # Compute the mean difference between A and B using this assignment. In
        # the swap array, simply map "0" -> +1 and "1" -> - 1.
        swap = [1 if x == "0" else -1 for x in assignment]
        diff = sum([(acc1[i] - acc2[i]) * swap[i] for i in range(n)]) / n
        # print(swap, diff, diff_observed)
        if abs(diff) >= abs(diff_observed):
            count += 1

    return count / num_samples


if __name__ == "__main__":
    if sys.argv != 2:
        print("Usage: python3 significance-tests.py <directory>")
        sys.exit(1)
