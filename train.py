from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


model = ResNet50(include_top=False, weights='imagenet', input_tensor=None)
print("loaded Resnet")

train_data_dir = '../pilz-scrapper/target/train'
validation_data_dir = '../pilz-scrapper/target/test'

def load_image(path="7.jpg"):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

x = load_image()
print("Loaded image")
features = model.predict(x)
print(features)