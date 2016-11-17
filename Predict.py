from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model
from datetime import datetime


resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

batch_size = 128
train_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/train'
test_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/test'
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

test_datagen = ImageDataGenerator(rescale=1./255)

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
print(class_dictionary)

lookupdict = {}
for key in class_dictionary.keys():
    value = class_dictionary[key]
    lookupdict[value] = key

print(lookupdict[367])

nb_categories = 1510

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_categories, activation='sigmoid')(x)
model = Model(input=resnet.input, output=predictions)


model.load_weights('weights/weights25l0.294604301453.hdf5')


model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")

img = load_img('tests/testimages/judasohr.jpg', target_size=image_size)
input = img_to_array(img)
input = input.reshape((1,) + input.shape)

print('predicting...')
output = model.predict(input)
print(output)
out = output.tolist()[0]
print(out)

highest = -1
highest_val = -1
for i in range(0, len(out)):
    if out[i] > highest_val:
        highest_val = out[i]
        highest = i

print(highest)
print(highest_val)
print(lookupdict[highest])



