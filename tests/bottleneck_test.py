import unittest
import bottleneck
from keras.applications.resnet50 import ResNet50
import os.path

class Bottleneck(unittest.TestCase):
    def test_load_image_folder(self):
        path = 'train/Birkenpilz/'
        array = bottleneck.load_image_folder(path)
        print(array.shape)
        self.assertEqual((6,3,224,224),array.shape)

    def test_save_bottleneck(self):
        path = 'train/Birkenpilz/'
        feature = path + "features.npy"
        if os.path.isfile(feature):
            os.remove(feature)
        model = ResNet50(include_top=False, weights='imagenet', input_tensor=None)
        bottleneck.save_bottleneck(path, model)
        exists = os.path.isfile(path + "/features.npy")
        self.assertTrue(exists)

    def test_save_bottlenecks(self):
        path = "train"
        feature1 = "train/Birkenpilz/features.npy"
        feature2 = "train/Eselsohr/features.npy"
        if os.path.isfile(feature1):
            os.remove(feature1)
        if os.path.isfile(feature2):
            os.remove(feature2)
        bottleneck.save_bottlenecks(path)
        exists = os.path.isfile(feature1)
        self.assertTrue(exists)
        exists = os.path.isfile(feature2)
        self.assertTrue(exists)


if __name__ == '__main__':
    unittest.main()
