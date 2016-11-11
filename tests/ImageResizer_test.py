import unittest
import ImageResizer


class FeatureLoadTest(unittest.TestCase):
    def cat_resize_test(self):
        source = "/home/severin/PycharmProjects/pilz-scrapper/target/test"
        destination = "/home/severin/PycharmProjects/pilz-scrapper/resized/test"
        ImageResizer.resize_dataset(source,destination)