import numpy as np


def num_parameters(estimator):
    variable_names = estimator.estimator.get_variable_names()
    n_params = 0
    n_skipped = 0
    for vn in variable_names:
        if "Adam" in vn:
            n_skipped += 1
        else:
            val = np.asarray(estimator.estimator.get_variable_value(vn))
            v_params = np.prod(val.shape, dtype=int)
            n_params += v_params
    print("Info: counting parameters: skipped %i variables whose name contains 'Adam'." % n_skipped)
    return n_params
