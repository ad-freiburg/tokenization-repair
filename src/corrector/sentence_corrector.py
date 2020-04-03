import numpy as np

from project import src
from src.encoding.one_hot import encode, labels
from src.models.unidirectional import UniDirectionalModel, UniDirectionalModelSpecification
from src.models.bidirectional import BidirectionalModel, BidirectionalModelSpecification


EPSILON = 1e-16


def elementwise_mean(a, b):
    return (a + b) / 2


def normalize(matrix):
    return (matrix.T / np.sum(matrix, axis=1)).T


class SentenceCorrector:
    def get_blank_insertion_probabilities(self, sequence, greedy=False, stop=0.1, maxi=10):
        if greedy:
            prob_dict = self.get_blank_insertion_probabilities_greedy(sequence, stop, maxi)
        else:
            probs = self.get_blank_insertion_probabilities_oneshot(sequence)
            prob_dict = {i: (probs[i], 0) for i in range(len(probs))}
        return prob_dict

    def get_blank_insertion_probabilities_greedy(self, sequence, stop, maxi):
        insertion_probs = {}
        original_positions = list(range(len(sequence) + 1))
        current_prob = 1
        inserts = 0
        while current_prob >= stop and inserts < maxi:
            probs = self.get_blank_insertion_probabilities_oneshot(sequence)
            amax = np.argmax(probs)
            orig_ix = original_positions[amax]
            maxprob = probs[amax]
            current_prob = min(maxprob, current_prob)
            if current_prob >= stop:
                if orig_ix != -1 and orig_ix not in insertion_probs:
                    insertion_probs[orig_ix] = (current_prob, inserts + 1)
                # insert
                sequence = sequence[:amax] + ' ' + sequence[amax:]
                original_positions = original_positions[:amax] + [-1] + original_positions[amax:]
                inserts += 1
        return insertion_probs

    def get_blank_insertion_probabilities_oneshot(self, sequence):
        raise NotImplementedError

    def get_blank_deletion_probabilities(self, sequence, greedy=False, stop=0.1, maxi=10):
        if greedy:
            prob_dict = self.get_blank_deletion_probabilities_greedy(sequence, stop=stop, maxi=maxi)
        else:
            prob_dict = self.get_blank_deletion_probabilities_oneshot(sequence)
            prob_dict = {i: (prob_dict[i], 0) for i in prob_dict}
        return prob_dict

    def get_blank_deletion_probabilities_oneshot(self, sequence):
        raise NotImplementedError

    def get_blank_deletion_probabilities_greedy(self, sequence, stop, maxi):
        original_positions = list(range(len(sequence)))
        probabilities = {}
        current = float("inf")
        iteration = 1
        while current >= stop and iteration <= maxi:
            del_probs = self.get_blank_deletion_probabilities_oneshot(sequence)
            if len(del_probs) == 0:
                break
            delete = max([(del_probs[i], i) for i in del_probs])
            current = min(current, delete[0])
            del_pos = delete[1]
            if current >= stop:
                probabilities[original_positions[del_pos]] = (current, iteration)
                sequence = sequence[:del_pos] + sequence[(del_pos + 1):]
                original_positions = original_positions[:del_pos] + original_positions[(del_pos + 1):]
            iteration += 1
        return probabilities

    def get_probabilities(self, sequence):
        raise NotImplementedError

    def sequence_log_likelihood(self, sequence):
        probs = self.get_probabilities(sequence)
        char_ix = labels(sequence, self.encoder_dict)[1:-1]
        seq_probs = [probs[i, char_ix[i]] for i in range(len(sequence))]
        log_probs = np.log(np.asarray(seq_probs) + EPSILON)
        log_likelihood = np.sum(log_probs)
        return log_likelihood

    def end(self):
        raise NotImplementedError

    def get_insertion_probabilities_oneshot(self, sequence):
        raise NotImplementedError

    def get_deletion_probabilities_oneshot(self, sequence):
        raise NotImplementedError


class UnidirectionalCorrector(SentenceCorrector):
    def __init__(self, model_path, encoder_dict, backward=False):
        self.model = UniDirectionalModel()
        self.model.load(model_path)
        self.encoder_dict = encoder_dict
        self.backward = backward

    def get_probabilities(self, sequence, include_last=False):
        encoded = encode(sequence, self.encoder_dict)
        if self.backward:
            encoded = encoded[::-1, :]
        if include_last:
            encoded = encoded[:-1, :]
        else:
            encoded = encoded[:-2, :]
        probs = self.model.predict(encoded)
        if self.backward:
            probs = probs[::-1, :]
        return probs


class BidirectionalCorrector(SentenceCorrector):
    def __init__(self, model, encoder_dict, decoder_dict=None, model_path=None):
        if model_path is not None:
            self.model = BidirectionalModel()
            self.model.load(model_path)
        else:
            self.model = model
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict

    def get_blank_insertion_probabilities_oneshot(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        insertion_probs = self.model.predict_insertion(encoded)
        blank_ix = self.encoder_dict[' ']
        blank_probs = insertion_probs[:, blank_ix]
        return blank_probs

    def get_probabilities(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        return self.model.predict(encoded)

    def get_blank_deletion_probabilities_oneshot(self, sequence):
        probs = self.get_probabilities(sequence)
        deletion_probs = {i: 1 - probs[i, self.encoder_dict[' ']] for i in range(len(sequence)) if sequence[i] == ' '}
        return deletion_probs

    def end(self):
        self.model.close_session()

    def get_all_insertion_probabilities(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        probs = self.model.predict_insertion(encoded)
        return probs

    def get_insertion_probabilities_oneshot(self, sequence):
        probs = self.get_all_insertion_probabilities(sequence)
        amax = np.argmax(probs, axis=1)
        insertion_probs = [probs[i, amax[i]] for i in range(len(amax))]
        insertion_chars = [self.decoder_dict[a] for a in amax]
        return insertion_probs, insertion_chars

    def get_deletion_probabilities_oneshot(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        sequence_labels = labels(sequence, self.encoder_dict)[1:-1]
        probs = self.model.predict(encoded)
        deletion_probs = [1 - probs[i, sequence_labels[i]] for i in range(len(sequence))]
        return deletion_probs


class CombinedCorrector(SentenceCorrector):
    def __init__(self, fwd_model, bwd_model, encoder_dict, method, fwd_path=None, bwd_path=None, decoder_dict=None):
        if fwd_path is None:
            self.fwd_model = fwd_model
        else:
            self.fwd_model = UniDirectionalModel()
            self.fwd_model.load(fwd_path)
        if bwd_path is None:
            self.bwd_model = bwd_model
        else:
            self.bwd_model = UniDirectionalModel()
            self.bwd_model.load(bwd_path)
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict
        self.method = method

    def combine_blank_probabilities(self, fwd_probs, bwd_probs, char_ix_after, char_ix_before):
        if self.method == "average":
            return self.get_blank_probabilities(fwd_probs, bwd_probs, operator=elementwise_mean)
        if self.method == "average_compared":
            return self.get_blank_probabilities_compared(fwd_probs, bwd_probs, char_ix_after, char_ix_before,
                                                         operator=elementwise_mean)
        if self.method == "multiply":
            return self.get_blank_probabilities(fwd_probs, bwd_probs)
        if self.method == "multiply_normalized":
            return self.get_blank_probabilities(fwd_probs, bwd_probs, normalized=True)
        if self.method == "multiply_compared":
            return self.get_blank_probabilities_compared(fwd_probs, bwd_probs, char_ix_after, char_ix_before)
        if self.method == "multiply_compared_min":
            return self.get_blank_probabilities_compared(fwd_probs, bwd_probs, char_ix_after, char_ix_before,
                                                         minimum=True)
        raise NotImplementedError("Combination method %s unknown." % self.method)

    def combine_probabilities(self, fwd_probs, bwd_probs, char_ix_after, char_ix_before):
        if self.method == "average":
            return self.get_character_probabilities(fwd_probs, bwd_probs, operator=elementwise_mean)
        if self.method == "average_compared":
            return self.get_character_probabilities_compared(fwd_probs, bwd_probs, char_ix_after, char_ix_before,
                                                             operator=elementwise_mean)
        if self.method == "multiply":
            return self.get_character_probabilities(fwd_probs, bwd_probs)
        if self.method == "multiply_normalized":
            return self.get_character_probabilities(fwd_probs, bwd_probs, normalized=True)
        if self.method == "multiply_compared":
            return self.get_character_probabilities_compared(fwd_probs, bwd_probs, char_ix_after, char_ix_before)
        raise NotImplementedError("Combination method %s unknown." % self.method)

    def get_blank_probabilities(self, fwd_probs, bwd_probs, operator=np.multiply, normalized=False):
        blank_ix = self.encoder_dict[' ']
        if not normalized:
            combined = operator(fwd_probs[:, blank_ix], bwd_probs[:, blank_ix])
        else:
            combined = operator(fwd_probs, bwd_probs)
            combined = normalize(combined)
            combined = combined[:, blank_ix]
        return combined

    def get_blank_probabilities_compared(self, fwd_probs, bwd_probs, char_ix_after, char_ix_before, operator=np.multiply,
                                         minimum=False):
        blank_ix = self.encoder_dict[' ']
        blank_prob = operator(fwd_probs[:, blank_ix], bwd_probs[:, blank_ix])
        sequence_prob = operator(np.array([fwd_probs[i, char_ix_after[i]] for i in range(len(blank_prob))]),
                                 np.array([bwd_probs[i, char_ix_before[i]] for i in range(len(blank_prob))]))
        if not minimum:
            combined = blank_prob / (blank_prob + sequence_prob)
        else:
            #combined = np.min(np.stack([blank_prob, 1 - sequence_prob]), axis=0)
            combined = blank_prob * (1 - sequence_prob)
        return combined

    def get_character_probabilities(self, fwd_probs, bwd_probs, operator=np.multiply, normalized=False):
        combined = operator(fwd_probs, bwd_probs)
        if normalized:
            combined = normalize(combined)
        return combined

    def get_character_probabilities_compared(self, fwd_probs, bwd_probs, char_ix_after, char_ix_before,
                                             operator=np.multiply):
        seq_len = fwd_probs.shape[0]
        char_prob = operator(fwd_probs, bwd_probs)
        sequence_prob = operator(np.array([fwd_probs[i, char_ix_after[i]] for i in range(seq_len)]),
                                 np.array([bwd_probs[i, char_ix_before[i]] for i in range(seq_len)]))
        combined = char_prob / (char_prob + sequence_prob[:, np.newaxis])
        return combined

    def get_blank_insertion_probabilities_oneshot(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        char_indices = labels(sequence, self.encoder_dict)
        fwd_probs = self.fwd_model.predict(encoded)
        bwd_probs = self.bwd_model.predict(encoded[::-1, :])
        blank_probs = self.combine_blank_probabilities(fwd_probs[:-1, :], bwd_probs[:-1, :][::-1, :], char_indices[1:],
                                                       char_indices[:-1])
        return blank_probs

    def get_probabilities(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        fwd_probs = self.fwd_model.predict(encoded)[:-2, :]
        bwd_probs = self.bwd_model.predict(encoded[::-1, :])[:-2, :][::-1, :]
        if self.method == "multiply":
            probs = fwd_probs * bwd_probs
        elif self.method == "multiply_normalized":
            probs = fwd_probs * bwd_probs
            probs = normalize(probs)
        elif self.method == "average":
            probs = (fwd_probs + bwd_probs) / 2
        else:
            raise NotImplementedError("Combination method %s unknown." % self.method)
        return probs

    def get_blank_deletion_probabilities_oneshot(self, sequence):
        blank_positions = [i for i in range(len(sequence)) if sequence[i] == ' ']
        encoded = encode(sequence, self.encoder_dict)
        y_labels = labels(sequence, self.encoder_dict)
        fwd_probs = self.fwd_model.predict(encoded)[:-2, :]
        bwd_probs = self.bwd_model.predict(encoded[::-1])[:-2, :][::-1, :]
        deletion_probs = 1 - self.combine_blank_probabilities(fwd_probs, bwd_probs, y_labels[2:], y_labels[:-2])
        return {i: deletion_probs[i] for i in blank_positions}

    def end(self):
        self.fwd_model.close_session()
        self.bwd_model.close_session()

    def get_all_insertion_probabilities(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        char_indices = labels(sequence, self.encoder_dict)
        fwd_probs = self.fwd_model.predict(encoded)
        bwd_probs = self.bwd_model.predict(encoded[::-1, :])
        combined_probs = self.combine_probabilities(fwd_probs[:-1, :], bwd_probs[:-1, :][::-1, :], char_indices[1:],
                                                    char_indices[:-1])
        return combined_probs

    def get_insertion_probabilities_oneshot(self, sequence):
        combined_probs = self.get_all_insertion_probabilities(sequence)
        amax = np.argmax(combined_probs, axis=1)
        insertion_chars = [self.decoder_dict[a] for a in amax]
        insertion_probs = [combined_probs[i, amax[i]] for i in range(len(sequence) + 1)]
        return insertion_probs, insertion_chars

    def get_deletion_probabilities_oneshot(self, sequence):
        encoded = encode(sequence, self.encoder_dict)
        y_labels = labels(sequence, self.encoder_dict)
        fwd_probs = self.fwd_model.predict(encoded)[:-2, :]
        bwd_probs = self.bwd_model.predict(encoded[::-1])[:-2, :][::-1, :]
        character_probs = self.combine_probabilities(fwd_probs, bwd_probs, y_labels[2:], y_labels[:-2])
        deletion_probs = 1 - np.array([character_probs[i, y_labels[i + 1]] for i in range(len(sequence))])
        return deletion_probs
