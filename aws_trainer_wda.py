from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from FeatureLoad import load_dataset
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model


resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

train_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/train'
test_data_dir = '/home/severin/PycharmProjects/pilztrainer/mushroom_dataset/test'
image_size = (224,224)
shift=0.2
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=64,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=16,
        class_mode='categorical')

class_dictionary = validation_generator.class_indices
print(class_dictionary)

nb_categories = 1510

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_categories, activation='sigmoid')(x)
model = Model(input=resnet.input, output=predictions)

for layer in resnet.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")

callbacks = [ModelCheckpoint("weights/weight{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                           save_best_only=True, save_weights_only=False, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            ]
for i in range(0, 500):
    print(i)
    x_train, y_train = train_generator.next()
    x_test, y_test = validation_generator.next()
    print('\ttrain ' + str(model.train_on_batch(x_train, y_train)))
    print('\ttest' + str(model.test_on_batch(x_test, y_test)))
    #model.fit_generator(train_generator, validation_data=validation_generator, samples_per_epoch=32,nb_epoch=200,callbacks=callbacks, nb_val_samples=33920)
