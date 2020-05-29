from constants import FIXERS_ENUM

class dummy:
    def fix(self, text):
        import numpy as np
        if np.random.random() < 0.1:
            return text[:10] + ' ' + text[10:]
        else:
            return text

    def __repr__(self):
        return 'dymmu'


def construct_and_load_fixer(config):
    # return dummy()
    if config.fixer in [FIXERS_ENUM.bicontext_fixer,
                        FIXERS_ENUM.unidirctional_fixer]:
        #from .rnn_bicontext_fixer import RNNBicontextTextFixer
        from .bicontext_fixer_modified import RNNBicontextTextFixer
        return RNNBicontextTextFixer(config)
    elif config.fixer == FIXERS_ENUM.dp_fixer:
        from .dp_baseline_fixer import DPFixer
        return DPFixer(config)
