import json

from src.helper.files import get_files, read_lines
from src.settings import paths


def get_files_depth_two(directory):
    """
    Returns the paths to all files from the directories in the given directory.
    :param directory: a directory with directories containing the wanted files
    :return: list of full paths to all files at depth two
    """
    subdirs = get_files(directory)
    files = []
    for subdir in sorted(subdirs):
        path = directory + "/" + subdir + "/"
        subdir_files = get_files(path)
        for file in sorted(subdir_files):
            files.append(path + file)
    return files


def get_article_jsons():
    """
    Reads all articles as jsons from an extracted Wikipedia dump.
    :return: iterator over article jsons
    """
    wiki_text_directory = paths.WIKI_DIR + "text/"
    json_files = get_files_depth_two(wiki_text_directory)
    for file in json_files:
        lines = read_lines(file)
        for line in lines:
            article = json.loads(line)
            yield article
