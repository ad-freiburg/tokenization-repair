import datetime
import os

from constants import ENCODER_DICT, DECODER_DICT, SOS, EOS, UNK, encode, decode
from utils.logger import logger
from utils.utils import extract_file_name, normalize_distribution, makedirs

import numpy as np


class CharacterModel:
    def __init__(self, config):
        self.alphabet_size = len(ENCODER_DICT)

    def __repr__(self):
        return self.model_repr

    def softmaxs_to_codes(self, char_vec_list):
        """
        Transform a list of softmax vectors into the corresponding codes array.

        :returns: The corresponding string of the char_vec_list
        """
        return np.argmax(char_vec_list, axis=-1).astype(np.uint8)

    def softmaxs_to_str(self, char_vec_list):
        """
        Transform a list of softmax vectors into the corresponding string.

        :returns: The corresponding string of the char_vec_list
        """
        return self.codes_to_str(self.softmaxs_to_codes(char_vec_list))

    def code_to_char(self, code):
        return decode([code])

    def codes_to_str(self, codes):
        return decode(codes)

    def char_code(self, char):
        """
        Get the 0-based code of a character

        :returns: The 0-based id of char in the alphabet or alphabet length
        """
        return encode(char)[0]

    def str_codes(self, string):
        """
        Transform string to array of character codes.

        :returns: array of integer codes
        """
        return np.array(encode(string), dtype=np.uint8)

    def sample_char(self, char_vec, temperature=1.0, rolls=2):
        """
        Get the character from character vector, by choosing a random character
        based on given probability vector

        :param list char_vec: softmax list of the character scores
        :rtype: triple
        :returns: (code of the character, character, matching score)
        """
        probas = normalize_distribution(char_vec,
                                        temperature=temperature,
                                        rolls=rolls)
        code = np.argmax(probas)
        return self.code_to_char(code)

    def compile(self):
        from keras.optimizers import Adam, SGD
        self.model.compile(loss=self.losses, metrics=self.metrics,
                           optimizer=Adam(lr=self.learning_rate, decay=self.decay_rate))
        if not self.inference:
            self.model.summary(line_length=180, print_fn=logger.output)
            logger.log_info('compiled', self.model.metrics_names)

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.model_save_path
        makedirs(save_path)
        self.model.save(save_path)
        logger.log_info(save_path, 'saved..')

    def load_model(self, model_load_path):
        if os.path.isdir(os.path.dirname(model_load_path)):
            path, fil, ext = extract_file_name(model_load_path)
            all_files = [
                os.path.join(path, f)
                for f in os.listdir(path) if f.endswith(fil + '.' + ext)]
            # logger.log_debug(model_load_path, all_files)
            if all_files:
                load_path = max(all_files)
                from keras.models import load_model
                from models.custom_layers import all_custom_layers
                self.model = load_model(
                    load_path, custom_objects=all_custom_layers)
                logger.log_report(load_path, 'loaded..', highlight=4)
                try:
                    self.initial_epoch = int(extract_file_name(load_path)[1][2:7])
                except ValueError:
                    pass
                return True
        logger.log_error(model_load_path, 'not found..', highlight=6)
        return False

    def take_context(self, codes, st, en):
        context = codes[max(st, 0): min(en, len(codes))]
        # assert en - st in [1, self.history_length], (st, en)
        if st < 0:
            context = np.append([self.char_code()] * (-st), context)
        elif en > len(codes):
            context = np.append(context, [self.char_code()] * (en - max(st, len(codes))))
        # else:
        #     assert st >= 0 and en <= len(codes)
        #     assert (len(context) == self.history_length or len(context) == 1) and\
        #         context.ndim == 1, (np.shape(context), context)
        return context

    def sample_analysis(self):
        pass

    def create_saver_callback(self, freq, eval_freq):
        # TODO: update based on batches only
        from keras.callbacks import Callback

        class Checkpoint(Callback):
            def __init__(self, freq, eval_freq, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.freq = freq
                self.eval_freq = eval_freq
                self.total_steps_so_far = 0

            def on_train_begin(self, logs={}):
                logger.start()

            def on_epoch_begin(_self, epoch, logs={}):
                _self.st = datetime.datetime.now()

            def on_batch_end(_self, batch, logs={}):
                _self.total_steps_so_far += 1
                if (_self.total_steps_so_far + 1) % 1000 == 0:
                    path, fil, ext = extract_file_name(self.model_save_path)
                    save_path = os.path.join(
                        path, 'batch%.7d-' % (
                            _self.total_steps_so_far + 1) + fil + '.' + ext)
                    self.save_model(save_path)

                    delimiter = ' - '
                    logger.log_info('\nbatch: %d%s%s' % (
                        _self.total_steps_so_far + 1, delimiter,
                        delimiter.join('%s: %.5f' % i for i in logs.items())))

                    logger.log_full_report_into_file(
                        self.model_save_path,
                        keep_log=True)

                if (_self.total_steps_so_far + 1) % 1500 == 0:
                    self.sample_analysis()

            def on_epoch_end(_self, epoch, logs={}):
                delimiter = ' - '
                time_delta = datetime.datetime.now() - _self.st
                logger.log_info('\nepoch: %d%s%s%stime taken: %s' % (
                    epoch + 1, delimiter,
                    delimiter.join('%s: %.8f' % i for i in logs.items()),
                    delimiter, time_delta))
                if (epoch + 1) % _self.freq == 0:
                    path, fil, ext = extract_file_name(self.model_save_path)
                    save_path = os.path.join(
                        path, 'it%.5d-' % (epoch + 1) + fil + '.' + ext)
                    self.save_model(save_path)
                    logger.log_full_report_into_file(self.model_save_path,
                                                     keep_log=True)
                if (epoch + 1) % _self.eval_freq == 0:
                    self.sample_analysis()

            def on_train_end(_self, logs={}):
                logger.log_full_report_into_file(self.model_save_path)

        return Checkpoint(freq, eval_freq)
