from typing import List

import json
import random
import re
import sys

import project
from src.helper.files import get_files, read_sequences, write_lines, make_directory
from src.settings import paths
from src.helper.pickle import load_object
from src.sequence.sentence_splitter import NLTKSentenceSplitter


def get_article_ids():
    training = load_object(paths.WIKI_TRAINING_ARTICLE_IDS)
    development = load_object(paths.WIKI_DEVELOPMENT_ARTICLE_IDS)
    test = load_object(paths.WIKI_TEST_ARTICLE_IDS)
    return training, development, test


def strip_all(texts: List[str]) -> List[str]:
    texts = [text.strip() for text in texts]
    texts = [text for text in texts if len(text) > 0]
    return texts


def split_article(text: str) -> List[str]:
    paragraphs = text.split('\n')
    paragraphs = strip_all(paragraphs)
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(NLTKSentenceSplitter.split(paragraph))
    sentences = strip_all(sentences)
    return sentences


FILTER_REGEX = re.compile(r" [.,;]( |$)|<|>|\"\"|\(\)| ' |\([,;]|colspan")


def filter_sentences(sentences: List[str]) -> List[str]:
    sentences = [sentence for sentence in sentences if not FILTER_REGEX.search(sentence)]
    sentences = [sentence for sentence in sentences if len(sentence) > 1]
    #sentences = [sentence for sentence in sentences if not sentence.isnumeric()]
    return sentences


def unify_quotation_marks(text: str) -> str:
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    return text


def unify_spacing(text: str) -> str:
    text = ' '.join(text.split())
    while '  ' in text:
        text.replace('  ', ' ')
    return text


def preprocess_sentence(sentence: str) -> str:
    sentence = unify_quotation_marks(sentence)
    sentence = unify_spacing(sentence)
    return sentence


if __name__ == "__main__":
    TRAINING = sys.argv[1] == "training"
    random.seed(42)
    base_dir = paths.WIKI_DIR + "text/"
    make_directory(paths.WIKI_SENTENCES_DIR)

    training_ids, development_ids, test_ids = get_article_ids()
    tuning_ids = set(random.sample(training_ids, 10000))
    training_ids = set(training_ids)
    development_ids = set(development_ids)
    test_ids = set(test_ids)
    evaluation_ids = development_ids.union(test_ids).union(tuning_ids)

    tuning_sentences, development_sentences, test_sentences = [], [], []

    if TRAINING:
        training_file = open(paths.WIKI_TRAINING_SENTENCES, 'w', encoding="utf8")

    for sub_dir in sorted(get_files(base_dir)):
        print(sub_dir)
        for file in sorted(get_files(base_dir + sub_dir)):
            path = base_dir + sub_dir + "/" + file
            for line in read_sequences(path):
                article = json.loads(line)
                id = article["id"]
                if not TRAINING and id in evaluation_ids:
                    sentences = split_article(article["text"])
                    sentences = filter_sentences(sentences)
                    if len(sentences) > 0:
                        selected_sentence = random.choice(sentences)
                        selected_sentence = preprocess_sentence(selected_sentence)
                        if id in tuning_ids:
                            tuning_sentences.append(selected_sentence)
                        elif id in development_ids:
                            development_sentences.append(selected_sentence)
                        else:
                            test_sentences.append(selected_sentence)
                elif TRAINING and id in training_ids:
                    sentences = split_article(article["text"])
                    sentences = filter_sentences(sentences)
                    sentences = [preprocess_sentence(sentence) for sentence in sentences]
                    training_file.write('\n'.join(sentences + [""]))

    if TRAINING:
        training_file.close()
    else:
        for sentence_list in (tuning_sentences, development_sentences, test_sentences):
            random.shuffle(sentence_list)
        write_lines(paths.WIKI_TUNING_SENTENCES, tuning_sentences)
        write_lines(paths.WIKI_DEVELOPMENT_SENTENCES, development_sentences)
        write_lines(paths.WIKI_TEST_SENTENCES, test_sentences)
