import cv2
import math

import pandas as pd
import numpy as np


def read_data(path):
    data = pd.read_csv(path)
    video = data['video']
    label = data['label']
    label = pd.get_dummies((label))
    label = label.values[0:]
    label = np.array([i for i in label])
    video = np.array([i for i in video])
    return video, label


def tach_anh(file_name, name1):
    count = 0
    cap = cv2.VideoCapture(file_name)
    frameRate = cap.get(5)
    while (cap.isOpened()):
        frameId = cap.get(1)
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            name = name1 + '/' + file_name.split('\\')[1] + '/' + file_name.split('\\')[2].split('.')[
                0] + '_frame%d.jpg' % count;
            count += 1
            cv2.imwrite(name, frame)

    cap.release()


oo = '/data/phucnq/data/video_classification/'

video_train, lalel_train = read_data('/data/phucnq/data/video_classification/file_csv/train.csv')
for name_video in video_train:
    print(name_video)
    name = oo + name_video
    tach_anh(name, 'train')

video_test, lalel_test = read_data('/data/phucnq/data/video_classification/file_csv/test.csv')
for name_video in video_test:
    name = oo + name_video
    tach_anh(name, 'test')
print(2)
video_vali, lalel_vali = read_data('/data/phucnq/data/video_classification/file_csv/validation.csv')
for name_video in video_vali:
    name = oo + name_video
    tach_anh(name, 'validation')

print(3)
