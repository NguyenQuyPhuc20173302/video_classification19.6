from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model

base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)
model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)


def extract(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features[0]
    return features


import os
from tensorflow.keras.utils import to_categorical


def take_image(name_file):
    train_Image = []
    DIR = name_file
    name_label = []
    n = 0
    for name in os.listdir(DIR):
        name_label.append(name)
        folder = os.path.join(DIR, name)
        for img in os.listdir(folder):
            label = np.array([n])
            path = os.path.join(folder, img)
            image = extract(path)
            train_Image.append([np.array(image), label])

        n += 1

    x = np.array([i[0] for i in train_Image])
    y = np.array([i[1] for i in train_Image])
    y = to_categorical(y)
    return x, y


x, y = take_image('train')
import pickle

with open('/data/phucnq/data/video_classification/x_train.pkl', 'wb') as fl:
    pickle.dump(x, fl)

with open('/data/phucnq/data/video_classification/y_train.pkl', 'wb') as fl:
    pickle.dump(y, fl)

x, y = take_image('test')
import pickle

with open('/data/phucnq/data/video_classification/x_test.pkl', 'wb') as fl:
    pickle.dump(x, fl)

with open('/data/phucnq/data/video_classification/y_test.pkl', 'wb') as fl:
    pickle.dump(y, fl)

x, y = take_image('validation')
import pickle

with open('/data/phucnq/data/video_classification/x_vali.pkl', 'wb') as fl:
    pickle.dump(x, fl)

with open('/data/phucnq/data/video_classification/y_vali.pkl', 'wb') as fl:
    pickle.dump(y, fl)
