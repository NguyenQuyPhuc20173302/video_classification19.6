import os
import numpy as np


# tạo folder label trong train, test, validation để chứa ảnh
def create_folder(name):
    file = 'UCF-101'
    os.mkdir(name)
    for label in os.listdir(file):
        folders = os.path.join(name, label)
        os.mkdir(folders)


create_folder('train')
create_folder('test')
create_folder('validation')
