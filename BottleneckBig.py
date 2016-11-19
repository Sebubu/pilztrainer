from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from datetime import datetime
import os
import pwd
import numpy as np



def get_username():
    return pwd.getpwuid(os.getuid())[0]


#resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

batch_size = 512
test_batch_size = 128
nb_train_data = 187639

if get_username() == 'severin':
    train_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/train'
    test_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/test'
else:
    train_data_dir = '/home/ubuntu/mushroom_dataset/train'
    test_data_dir = '/home/ubuntu/mushroom_dataset/test'

image_size = (224, 224)
shift = 0.2
'''
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    width_shift_range=shift,
    height_shift_range=shift
)
'''
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir='aug/'
)


test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir='aug/'
)

resnet.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")

target = int(nb_train_data/batch_size)
for i in range(0,target):
    print(i, "/", target)
    x_train, y_train = test_generator.next()
    prediction = resnet.predict(x_train, batch_size)
    name = "i" + str(i)
    np.save('bottleneck/x3' + name + '.npy', prediction)
    np.save('bottleneck/y3' + name + '.npy', y_train)