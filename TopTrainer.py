from BottleneckLoader import load
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2, activity_l2

x_batch,y_batch = load()
x_test,y_test = load('bottlenecktest/')

print('train ')
print(x_batch.shape)
print(y_batch.shape)

print('test ')
print(x_test.shape)
print(y_test.shape)

nb_categories = 1510


inputs = Input(x_batch.shape[1:])
x = GlobalAveragePooling2D()(inputs)
x = Dense(1024)(x)
x = LeakyReLU()(x)
x = Dropout(0.5)(x)
predictions = Dense(y_batch.shape[1], activation='softmax')(x)
model = Model(input=inputs, output=predictions)

print('compile')
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [ModelCheckpoint("weights/weight{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                           save_best_only=True, save_weights_only=True, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            ]

print('fit')
model.fit(x_batch, y_batch, batch_size=2096, nb_epoch=100, shuffle=True, validation_data=(x_test,y_test))