from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from FeatureLoad import load_dataset
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU


model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

train_data_dir = '../pilz-scrapper/target/train'
test_data_dir = '../pilz-scrapper/target/test'

train_x, train_y = load_dataset(train_data_dir)
test_x, test_y = load_dataset(test_data_dir)
#train_x, train_y = load_dataset(test_data_dir)
#test_x, test_y = load_dataset(test_data_dir)
nb_categories = 208

top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(512))#, activation='relu'))
top_model.add(LeakyReLU())
top_model.add(Dropout(0.6))
top_model.add(Dense(nb_categories, activation='sigmoid'))


top_model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")

callbacks = [ModelCheckpoint("weights/weight{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                           save_best_only=True, save_weights_only=False, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            ]

top_model.fit(train_x, train_y, batch_size=32, nb_epoch=200,
          verbose=1, validation_data=(test_x, test_y), callbacks=callbacks)