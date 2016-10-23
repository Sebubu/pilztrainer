from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from os import listdir
from os.path import isdir, join, isfile
import datetime



def load_image(path="7.jpg"):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_image_folder(path):
    arrays = []
    image_paths = [join(path, fe) for fe in listdir(path) if isfile(join(path, fe))]
    for image_path in image_paths:
        if image_path.endswith(".npy"):
            continue
        array = load_image(image_path)
        arrays.append(array)
    if len(arrays) == 0:
        print(path)
    return np.concatenate(tuple(arrays))

def save_bottleneck(categorie_path, model):
    if isfile(categorie_path + "/features.npy"):
        return
    x = load_image_folder(categorie_path)
    features = model.predict(x)
    np.save(categorie_path + "/features", features)

def save_bottlenecks(path):
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=None)
    print("loaded Resnet")
    onlydir = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    i = 0
    for categorie in onlydir:
        save_bottleneck(categorie, model)
        i+=1
        print(str(datetime.datetime.now()) + ": " + str(i) + "/" + str(len(onlydir)))




train_data_dir = '../pilz-scrapper/target/train'
validation_data_dir = '../pilz-scrapper/target/test'

save_bottlenecks(validation_data_dir)
save_bottlenecks(train_data_dir)