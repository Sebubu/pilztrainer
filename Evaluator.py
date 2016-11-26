from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import os
import pwd
from keras.callbacks import ModelCheckpoint


def get_username():
    return pwd.getpwuid(os.getuid())[0]


resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

batch_size = 8

if get_username() == 'severin':
    train_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/train'
    test_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/test'
else:
    train_data_dir = '/home/ubuntu/mushroom_dataset/train'
    test_data_dir = '/home/ubuntu/mushroom_dataset/test'

image_size = (224, 224)


test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

class_dictionary = validation_generator.class_indices
tuple_list = zip(class_dictionary.values(), class_dictionary.keys())
lookup_dict = dict(tuple_list)
print(class_dictionary)
print(lookup_dict)

nb_categories = 1510

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)

predictions = Dense(nb_categories, activation='softmax')(x)
model = Model(input=resnet.input, output=predictions)


from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
print("Compiled")

model.load_weights('../xxWeight05-2.89.hdf5')
print('weights loaded')

def max_index(prediction):
    max = 0
    index = -1
    for i, liklyhood in enumerate(prediction):
        if max < liklyhood:
            index = i
            max = liklyhood
    return index

nb_positiv = 0
nb_negativ = 0
results = {}
no_match = {}


def count(x_t, y_t):
    global nb_positiv
    global nb_negativ
    prediction = model.predict_on_batch(x_t)
    for i, p in enumerate(prediction.tolist()):
        index = max_index(p)
        should_index = max_index(y_t[i])

        name = lookup_dict[should_index]
        if name not in results:
            results[name] = 0
        if name not in no_match:
            no_match[name] = 0

        if index == should_index:
            results[name] += 1
            nb_positiv += 1
        else:
            no_match[name] += 1
            nb_negativ += 1


nb_iterations = int(validation_generator.nb_sample/batch_size)
for i, data in enumerate(validation_generator):
    x_t, y_t = data
    print(i, "/", nb_iterations)
    count(x_t, y_t)
    print(nb_positiv/(nb_positiv + nb_negativ), "%")
    #print(results)
    if i > nb_iterations:
        break


print(results)
print(no_match)
print(nb_positiv/(nb_positiv + nb_negativ), "%")


