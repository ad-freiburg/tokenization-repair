#!/usr/bin/env python3
import argparse
import sys
import time
import multiprocessing
import warnings

from src.noise.acl_noise_inducer import ACLNoiseInducer
warnings.filterwarnings("ignore")


def corrupt_sequence_function(sequence):
    global noise_inducer
    return noise_inducer.induce_noise(sequence)


def corrupt_all(path, outpath, take=None, freq=10000):
    total = 108068848 if take is None else take
    last_out = ''
    with open(path, 'r') as src_file, open(outpath, 'w') as out_file:
        with multiprocessing.Pool(12) as pool:
            tic = time.time()
            #for idx, corrupt in enumerate(map(corrupt_sequence_function, src_file)):
            for idx, corrupt in enumerate(pool.imap(corrupt_sequence_function, src_file)):
                out_file.write(corrupt)  # print(corrupt, file=sys.stderr)
                if take is not None and idx >= take:
                    break
                if (idx % freq == freq - 1 or (take is not None and idx + 1 >= take)
                        or idx + 1 >= total):
                    toc = time.time()
                    rate = (toc - tic) / (idx + 1)
                    remain = int((total - idx - 1) * rate)
                    hour = '%.2d' % (remain // 3600)
                    remain = remain % 3600
                    minute = '%.2d' % (remain // 60)
                    seconds = '%.2d' % (remain % 60)
                    rate = '%.2f' % (rate * 1e6)
                    out = f'[{idx + 1}/{total}] {hour}:{minute}:{seconds} {rate} us/ex'

                    bs = len(last_out)
                    print('\b' * bs + ' ' * bs + '\b' * bs + out, end='',
                          flush=True)
                    last_out = out

        pool.join()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src-path', help='source dataset path',
        default='/nfs/datasets/tokenization-repair/training_mixed.txt'
    )
    parser.add_argument(
        '--dest-path', help='destination dataset path',
        default='/nfs/students/mostafa-mohamed/tokenization-repair-paper/training_ocr_2nd.txt'
    )
    args = parser.parse_args()

    #outpath = '/nfs/students/mostafa-mohamed/tokenization-repair-paper/training_ocr.txt'

    noise_inducer = ACLNoiseInducer(p=0.1, insertion_prob=0.2079, seed=42)
    corrupt_all(args.src_path, args.dest_path)
