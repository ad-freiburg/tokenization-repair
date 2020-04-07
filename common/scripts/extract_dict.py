#!/usr/bin/env python3
import os
import multiprocessing

from tqdm import tqdm

from configs import get_dataset_config
from constants import NUM_THREADS
from handlers.reader import Reader
from utils.logger import logger
from utils.utils import get_vocab, take_first_n


if __name__ == '__main__':
    config = get_dataset_config()
    reader = Reader(config)
    total, gen = reader.read_train_lines()

    vocab = {}
    with multiprocessing.Pool(NUM_THREADS) as pool:
        for cnt_dict in tqdm(pool.imap(get_vocab, gen), total=total):
            for word, cnt in cnt_dict.items():
                vocab[word] = vocab.get(word, 0) + cnt
        pool.close()
        pool.join()
    vocab = sorted([(word, cnt) for word, cnt in vocab.items() if cnt > 2])

    logger.log_report("writing into:", config.vocab_path)
    with open(config.vocab_path, 'w') as fl:
        for word, cnt in tqdm(vocab):
            fl.write("%s\t%d\n" % (word, cnt))
    logger.log_report("done.. wrote all into:", config.vocab_path)
