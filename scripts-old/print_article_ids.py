import sys

from project import src
from src.helper.pickle import load_object
from src.settings import paths


if __name__ == "__main__":
    if "test" in sys.argv:
        article_ids = load_object(paths.WIKI_TEST_ARTICLE_IDS)
    else:
        article_ids = load_object(paths.WIKI_DEVELOPMENT_ARTICLE_IDS)
    article_ids = [int(i) for i in article_ids]
    for i in sorted(article_ids):
        print(i)
