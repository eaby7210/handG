# coding: utf-8

# In[ ]:
import os

import tensorflow as tf
# sess = tf.Session()

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables

#------------------------------

import os, cv2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2

# import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')


def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    # res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))
def predictcnn(fn):
    dataset=read_dataset1(fn)
    (mnist_row, mnist_col, mnist_color) = 128, 128, 1
    # (mnist_row, mnist_col, mnist_color) = 48, 48, 1

    dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
    dataset /= 255
    mo = load_model(r"C:\Users\miche\PycharmProjects\os_controlling\model_new.h5")

    # predict probabilities for test set

    yhat_classes = mo.predict_classes(dataset, verbose=0)
    return yhat_classes

# labels_list=["screenshots", "volume down", "volume up"]
# res=predictcnn(r"C:\Users\miche\PycharmProjects\os_controlling\dataSet\training_data\volume up\1.jpg")
# print("RR  :  ", labels_list[res[0]])