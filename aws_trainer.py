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

train_data_dir = 'resized/train'
test_data_dir = 'resized/test'
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
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='binary')

class_dictionary = validation_generator.class_indices
print(class_dictionary)

nb_categories = 208

x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(512)(x)
x = LeakyReLU()(x)
x = Dropout(0.6)(x)
predictions = Dense(nb_categories, activation='sigmoid')(x)
model = Model(input=resnet.input, output=predictions)

for layer in resnet.layers:
    layer.trainable = False

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")

callbacks = [ModelCheckpoint("weights/weight{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                           save_best_only=True, save_weights_only=False, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            ]

model.fit_generator(train_generator, validation_data=validation_generator, samples_per_epoch=32,nb_epoch=200,callbacks=callbacks, nb_val_samples=1969)
