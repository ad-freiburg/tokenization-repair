import numpy as np
import keras
print(keras.__version__)

from configs import get_language_model_config
from models.rnn_language_model import RNNLanguageModel
from constants import SOS, ENCODER_DICT


if __name__ == '__main__':
    model = RNNLanguageModel(get_language_model_config(perturbate=0))
    s = "Hello world"
    #res = model.predict_seq(s)
    #ws = model.model.get_layer('RNN').get_weights()
    #i = 1
    #ws[-1][1024 * i : 1024 *(i + 1)] += 1
    #model.model.get_layer('RNN').set_weights(ws)
    codes = np.array([[SOS] + model.str_codes(s).tolist()])
    print(codes)
    print(codes.shape)
    logits = model.model.predict([codes] + model.new_state())[0][0]
    print([np.shape(x) for x in logits])
    print(logits.shape)
    for idx, code in enumerate(codes[0].tolist()[1:]):
        print(idx, code, float(logits[idx, code]))

    ps, state0, state1 = model.model.predict([np.array([[SOS, model.char_code('H')]])] + model.new_state())
    print(state0.tolist()[0][:10])
    print('--')
    print(state1.tolist()[0][:10])
    #for code in codes:
    #for code, (logits, _, _) in zip(codes, res):
    #    print(np.round(logits[0, code], 5))

