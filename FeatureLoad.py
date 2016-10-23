import numpy as np

from os import listdir
from os.path import isdir, join, isfile


def create_arr(number, categorie_count):
    zeros = np.zeros((1,categorie_count))
    zeros[0,number -1] = 1
    return zeros


def define_y(path):
    onlydir = [f for f in listdir(path) if isdir(join(path, f))]
    cat_count = len(onlydir)
    i = 1
    dict = {}
    for cat in onlydir:
        arr = create_arr(i,cat_count)
        dict[cat] = arr
        i+=1
    return dict


def save_y_dict(path):
    dict = define_y(path)
    np.save(path + '/y_dict.npy', dict)


def load_y_dict(path):
    return np.load(path + '/y_dict.npy').item()


def load_categorie_features(path):
    feature_file = path + "/features.npy"
    array = np.load(feature_file)
    return array


def create_y(base_array, times):
    y_list = []
    for i in range(0, times):
        y_list.append(base_array)

    return np.concatenate(tuple(y_list))


def load_dataset(path):
    dict = load_y_dict(path)
    categories = [f for f in listdir(path) if isdir(join(path, f))]
    all_features = []
    ys = []
    for cat in categories:
        cat_path = join(path, cat)

        features = load_categorie_features(cat_path)
        all_features.append(features)

        images_count = features.shape[0]
        single_y = dict[cat]
        y = create_y(single_y, images_count)
        ys.append(y)

    x = np.concatenate(tuple(all_features))
    y = np.concatenate(tuple(ys))

    return x, y



