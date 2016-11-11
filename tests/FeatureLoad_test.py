import unittest
import FeatureLoad


class FeatureLoadTest(unittest.TestCase):
    def test_CatFeatureLoad(self):
        path = 'train/Birkenpilz/'
        features = FeatureLoad.load_categorie_features(path)
        self.assertIsNotNone(features)

    def test_define_y(self):
        path = "../../pilz-scrapper/target/train"
        dict = FeatureLoad.define_y(path)
        FeatureLoad.save_y_dict(path)
        loaded_dict = FeatureLoad.load_y_dict(path)
        self.assertEqual(dict.keys(), loaded_dict.keys())


    def create_arr_test(self):
        arr = FeatureLoad.create_arr(2,3)
        self.assertEqual(arr[0].tolist(), [0,  1,  0])

    def load_dataset_test(self):
        path = "train"
        x, y = FeatureLoad.load_dataset(path)
        self.assertEqual(x.shape, (10, 2048, 1, 1))
        self.assertEqual(y.shape, (10, 2))

if __name__ == '__main__':
    unittest.main()
