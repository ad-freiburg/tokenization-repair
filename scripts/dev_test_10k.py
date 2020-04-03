import random

from project import src
from src.helper.pickle import load_object
from src.helper.files import write_lines
from src.settings import paths
from src.data.raw_wikipedia import get_article_jsons
from src.data.preprocessing import preprocess_sequence


def select_random_paragraph(text: str) -> str:
    paragraphs = text.split('\n')
    paragraphs = [preprocess_sequence(paragraph) for paragraph in paragraphs]
    paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) > 0]
    selected = random.choice(paragraphs)
    return selected


if __name__ == "__main__":
    random.seed(1998)

    development_ids = set(load_object(paths.WIKI_DEVELOPMENT_ARTICLE_IDS))
    test_ids = set(load_object(paths.WIKI_TEST_ARTICLE_IDS))
    print(development_ids)
    print(test_ids)

    development_paragraphs = []
    test_paragraphs = []

    for article in get_article_jsons():
        id = article["id"]
        is_dev = id in development_ids
        is_test = (not is_dev) and id in test_ids
        if is_dev or is_test:
            paragraph = select_random_paragraph(article["text"])
            if is_dev:
                development_paragraphs.append(paragraph)
            elif is_test:
                test_paragraphs.append(paragraph)
            print("%i dev, %i test" % (len(development_paragraphs), len(test_paragraphs)))

    random.shuffle(development_paragraphs)
    random.shuffle(test_paragraphs)

    write_lines(paths.WIKI_DEVELOPMENT_FILE, development_paragraphs)
    write_lines(paths.WIKI_TEST_FILE, test_paragraphs)
