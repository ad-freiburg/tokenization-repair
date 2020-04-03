import unittest

from src.helper.files import parent_directory


class FilesHelperTest(unittest.TestCase):
    def test_parent_directory(self):
        dir = parent_directory("/a/b/c")
        self.assertEqual("/a/b/", dir)
        dir = parent_directory("/a/b/c/")
        self.assertEqual("/a/b/", dir)


if __name__ == "__main__":
    unittest.main()
