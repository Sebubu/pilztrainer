from keras.preprocessing.image import ImageDataGenerator
from os.path import join
from os import mkdir
from distutils.dir_util import copy_tree
from shutil import rmtree
from datetime import datetime


train_data_dir = '/home/severin/Downloads/mushroom_dataset/train/'
temp_dataset = 'images/dataset'
name = 'Abortiporus biennis'

def create_temp_dataset(mushroom_name):
    mush_path = join(train_data_dir, mushroom_name)
    rmtree(temp_dataset)
    mkdir(temp_dataset)
    target_dir = join(temp_dataset, mushroom_name)
    copy_tree(mush_path, target_dir)


def generate(amount=32):
    image_size = (224,224)
    shift=0.2

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

    train_generator = train_datagen.flow_from_directory(
           temp_dataset,
            target_size=image_size,
            batch_size=amount,
            class_mode='categorical',
            save_to_dir='images/aug', save_prefix='aug', save_format='jpg')


    count = 0
    for X_batch, y_batch in train_generator:
        count += len(X_batch)
        if count > amount:
            break

print(datetime.now())

#create_temp_dataset(name)
generate()

print(datetime.now())