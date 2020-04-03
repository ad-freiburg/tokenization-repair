from src.helper.files import path_exists, make_directory


PARAGRAPH_ID_LEN = 8
SUB_SEQUENCE_ID_LEN = 4
SUBSPLIT_ID_LEN = 4
MAX_TITLE_LEN = 20


class BenchmarkFileNameGenerator:
    def __init__(self, out_directory, benchmark_name, dev_ids, test_ids, folder_size=5000):
        if not out_directory.endswith("/"):
            out_directory += "/"
        self.out_directory = out_directory
        self.benchmark_name = benchmark_name
        self.dev_ids = dev_ids
        self.test_ids = test_ids
        self.folder_size = folder_size
        self.n_training_files = 0
        self.n_development_files = 0
        self.n_test_files = 0
        self.n_paragraphs = 0

    def prepare_directories(self):
        for split in ["training", "development", "test"]:
            split_path = self.out_directory + split
            if not path_exists(split_path):
                make_directory(split_path)
            benchmark_split_path = split_path + "/" + self.benchmark_name
            if not path_exists(benchmark_split_path):
                make_directory(benchmark_split_path)
            texts_path = benchmark_split_path + "/texts"
            if not path_exists(texts_path):
                make_directory(texts_path)

    def get_sequence_files(self, article, n_article_paragraphs):
        files = []
        for sequence_ix in range(n_article_paragraphs):
            if article["id"] in self.dev_ids:
                split_name = "development"
                subsplit = self.n_development_files // self.folder_size
                self.n_development_files += 1
            elif article["id"] in self.test_ids:
                split_name = "test"
                subsplit = self.n_test_files // self.folder_size
                self.n_test_files += 1
            else:
                split_name = "training"
                subsplit = self.n_training_files // self.folder_size
                self.n_training_files += 1
            subsplit = ("%." + str(SUBSPLIT_ID_LEN) + "i") % subsplit
            folder = self.out_directory + split_name + "/" + self.benchmark_name + "/texts/" + subsplit + "/"
            if not path_exists(folder):
                make_directory(folder)
            file_name_pattern = "%." + str(PARAGRAPH_ID_LEN) + "i_%s_%s_%." + str(SUB_SEQUENCE_ID_LEN) + "i.txt"
            file_name = file_name_pattern % (self.n_paragraphs,
                                             article["id"],
                                             article["title"][:MAX_TITLE_LEN].replace('/', '_'),
                                             sequence_ix)
            files.append(folder + file_name)
            self.n_paragraphs += 1
        return files
