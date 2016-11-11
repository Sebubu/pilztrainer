from keras.applications.resnet50 import ResNet50
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Input
import bottleneck
import FeatureLoad

from keras.layers.advanced_activations import LeakyReLU

weights = "weights/512_06_leaky_weight18-2.14.hdf5"

nb_categories = 208

model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(3, 224, 224)))
print("loaded Resnet")

top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=model.output_shape[1:]))
#top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(512))#, activation='relu'))
top_model.add(LeakyReLU())
top_model.add(Dropout(0.6))
top_model.add(Dense(nb_categories, activation='sigmoid'))

top_model.load_weights(weights)

dict = FeatureLoad.load_y_dict('../pilz-scrapper/target/train')
tured_dict = {}

def get_number(arr):
    i = 0
    for val in arr[0].tolist():
        i+=1
        if val == 1:
            return i

for key in dict.keys():
    val = dict[key]
    number = get_number(val)
    tured_dict[number] = key

print(tured_dict)

def predict(path):
    x = bottleneck.load_image(path)
    features = model.predict(x)
    output = top_model.predict(features)
    return output
    l = output[0].tolist()
    number = -1
    max = 0
    sum = 0
    for i in range(0,210):
        sum +=  l[i]
        if max < l[i]:
            max = l[i]
            number = i + 1

    print(max)
    print(number)
    print(sum)
    print(tured_dict[number])




def top5(prediction):
    l = prediction[0].tolist()
    number = 0
    top5 = Top5()
    for i in range(0,208):
        number = i + 1
        val = l[i]
        top5.add(val, number)
    print(top5.calc_top5())


class Top5:
    def __init__(self):

        self.entries = {}

    def add(self, value, number):
        self.entries[value] = number

    def calc_top5(self):
        od = sorted(self.entries.items())
        top = []
        for i in range (-5, 0):
            val, number = od[i]
            print(val)
            name = tured_dict[number]
            top.append((val,name))
        return list(reversed(top))

while(True):
    path = input("Enter image path: tests/testimages/")
    print("Classify...")
    prediction = predict("tests/testimages/" + path)
    top5(prediction)