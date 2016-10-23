from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.models import Sequential
from FeatureLoad import load_dataset
from keras.layers import Input


model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")
print(model.output_shape[1:])

train_data_dir = '../pilz-scrapper/target/train'
test_data_dir = '../pilz-scrapper/target/test'

#train_x, train_y = load_dataset(test_data_dir)
#test_x, test_y = load_dataset(test_data_dir)
train_x, train_y = load_dataset("tests/train")
test_x, test_y = load_dataset("tests/train")
nb_categories = 2#210

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_categories, activation='sigmoid'))


top_model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
print("Compiled")

top_model.fit(train_x, train_y, batch_size=32, nb_epoch=200,
          verbose=1, validation_data=(test_x, test_y))