from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from datetime import datetime
import os
import pwd


def get_username():
    return pwd.getpwuid(os.getuid())[0]


#resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
whole_resnet = ResNet50(include_top=False, weights=None, input_tensor=Input(shape=(3, 224, 224)))
resnet = Model(input=whole_resnet.input, output=whole_resnet .get_layer('activation_46').output)
print("loaded Resnet")

batch_size = 512
test_batch_size = 128

if get_username() == 'severin':
    train_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/train'
    test_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/test'
else:
    train_data_dir = '/home/ubuntu/mushroom_dataset/train'
    test_data_dir = '/home/ubuntu/mushroom_dataset/test'

image_size = (224, 224)
shift = 0.2
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


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=test_batch_size,
    class_mode='categorical')

class_dictionary = validation_generator.class_indices

nb_categories = 1510

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
x = Dense(2048)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_categories, activation='softmax')(x)
model = Model(input=resnet.input, output=predictions)

for layer in resnet.layers:
    layer.trainable = False


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")


def printen(titel, result):
    loss = result[0]
    acc = result[1]
    print("\t" + titel + " loss " + str(loss) + ", acc " + str(acc))

loss = 100
for i in range(0, 500):
    print()
    start = datetime.now()
    print("Epoche " + str(i))
    x_train, y_train = train_generator.next()
    x_test, y_test = validation_generator.next()

    train_results = model.train_on_batch(x_train, y_train)
    printen("train", train_results)

    test = model.test_on_batch(x_test, y_test)
    printen("test", test)
    end = datetime.now()
    eta = end-start
    print("\t" + str(eta.seconds) + " seconds")

    test_loss = test[0]
    if loss > test_loss:
        print("\tsave model...")
        loss = test_loss
        model.save_weights('weights/weights' + str(i) + 'l' + str(test_loss) + '.hdf5')

model.save_weights('weights/weights500Finito.hdf5')
