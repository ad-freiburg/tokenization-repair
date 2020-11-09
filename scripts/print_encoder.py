import sys

from project import src
from src.encoding.character_encoder import get_encoder, get_arxiv_encoder


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "arxiv":
        print(get_arxiv_encoder().encoder)
    else:
        print(get_encoder(n=200).encoder)
