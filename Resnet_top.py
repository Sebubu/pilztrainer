from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import os
import pwd
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint


def topx(k):
    def topfunc(y_true, y_pred, k=k):
        return top_k_categorical_accuracy(y_true, y_pred, k)
    topfunc.__name__ = "top" + str(k)
    return topfunc


def get_username():
    return pwd.getpwuid(os.getuid())[0]


resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

batch_size = 512

if get_username() == 'severin':
    train_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/train'
    test_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/test'
else:
    train_data_dir = '/home/ubuntu/mushroom_dataset/train'
    test_data_dir = '/home/ubuntu/mushroom_dataset/test'

shift_range = 0.1

image_size = (224, 224)
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    #width_shift_range=shift_range,
    #height_shift_range=shift_range,
    #zoom_range=0.2
)


test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

class_dictionary = validation_generator.class_indices

nb_categories = 1510

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)

predictions = Dense(nb_categories, activation='softmax')(x)
model = Model(input=resnet.input, output=predictions)

'''
layer -1: 163
layer -2: 153
layer -3: 141
layer -4: 131
'''

for i, layer in enumerate(resnet.layers[:153]):
    layer.trainable = False

for i, layer in enumerate(resnet.layers):
    trainable = False
    if hasattr(layer, 'trainable'):
        trainable = layer.trainable
    print(i, layer.name, '\t', trainable)

from keras.optimizers import Adadelta
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(lr=1),
              metrics=['accuracy', topx(3), topx(5)])
print("Compiled")

model.load_weights('weights/xxWeight81-2.76.hdf5')
print('weights loaded')

callbacks = [ModelCheckpoint("weights/xxWeight{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                           save_best_only=True, save_weights_only=True, mode='auto')
            ]


model.fit_generator(train_generator,samples_per_epoch=batch_size*40, nb_epoch=500,
                    validation_data=validation_generator,nb_val_samples=batch_size*5,
                    callbacks=callbacks)

model.save_weights('weights/finishe.hdf5')



