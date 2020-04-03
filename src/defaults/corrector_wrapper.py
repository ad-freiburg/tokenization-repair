
from src.corrector.sentence_corrector import BidirectionalCorrector, CombinedCorrector
from src.corrector.greedy_sentence_corrector_wrapper import GreedySentenceCorrectorWrapper
from src.data.simplewiki import Simplewiki
from src.settings import paths

from bidirectional_bs import BidirectionalModel
from unidirectional_bs import UniDirectionalModel


def get_default_bidirectional_corrector_wrapper(insertion_threshold=0.837327,
                                                deletion_threshold=0.994371,
                                                sequence_probability=False):
    model = BidirectionalModel()
    model.load(paths.BIDIRECTIONAL_MODEL_DIR)
    encoder, decoder = Simplewiki.get_dictionaries()
    corrector = BidirectionalCorrector(model, encoder, decoder_dict=decoder)
    wrapper = GreedySentenceCorrectorWrapper(corrector,
                                             corrector,
                                             insertion_threshold=insertion_threshold,
                                             deletion_threshold=deletion_threshold,
                                             eval_sequence_probability=sequence_probability)
    return wrapper


def get_default_mixed_corrector_wrapper():
    encoder, decoder = Simplewiki.get_dictionaries()
    inserter = BidirectionalCorrector(None, encoder, model_path=paths.BIDIRECTIONAL_MODEL_DIR,
                                      decoder_dict=decoder)
    deleter = CombinedCorrector(None, None, encoder, "multiply_compared",
                                fwd_path=paths.UNIDIRECTIONAL_FWD_DIR,
                                bwd_path=paths.UNIDIRECTIONAL_BWD_DIR,
                                decoder_dict=decoder)
    corrector_wrapper = GreedySentenceCorrectorWrapper(inserter, deleter,
                                                       insertion_threshold=0.833474,
                                                       deletion_threshold=0.383286)
    return corrector_wrapper
