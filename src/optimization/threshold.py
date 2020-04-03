import numpy as np

from src.helper.data_structures import unique_on_sorted


def optimal_f1_threshold(true_probabilities, false_probabilities):
    true_probabilities = sorted(true_probabilities, reverse=True)
    false_probabilities = sorted(false_probabilities, reverse=True)
    total_true = len(true_probabilities)
    thresholds = unique_on_sorted(true_probabilities)
    tp_vec = [0] * len(thresholds)
    fp_vec = [0] * len(thresholds)
    true_pointer = 0
    false_pointer = 0
    for t_i, threshold in enumerate(thresholds):
        while true_pointer < len(true_probabilities) and true_probabilities[true_pointer] >= threshold:
            true_pointer += 1
        tp_vec[t_i] = true_pointer
        while false_pointer < len(false_probabilities) and false_probabilities[false_pointer] >= threshold:
            false_pointer += 1
        fp_vec[t_i] = false_pointer
    #print("thresholds")
    #print(thresholds)
    tp_vec = np.array(tp_vec)
    #print("tp")
    #print(tp_vec)
    fp_vec = np.array(fp_vec)
    #print("fp")
    #print(fp_vec)
    prec = tp_vec / (tp_vec + fp_vec)
    #print("prec")
    #print(prec)
    rec = tp_vec / total_true
    #print("rec")
    #print(rec)
    f1 = 2 * prec * rec / (prec + rec)
    #print("f1")
    #print(f1)
    best = int(np.argmax(f1))
    print("best f1 = %.4f @ %.6f (precision = %.4f, recall = %.4f)" % (
        f1[best], thresholds[best], prec[best], rec[best]
    ))
    return thresholds[best]
