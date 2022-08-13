import os
import sys
from cv2 import cv2
import numpy as np
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array


def import_data():
    base_dir = "D:/Machine learning projects/Datasets/Face Expression/images/train/"
    angry_images = sorted(list(paths.list_images(base_dir + "angry/")))
    disgust_images = sorted(list(paths.list_images(base_dir + "disgust/")))
    fear_images = sorted(list(paths.list_images(base_dir + "fear/")))
    happy_images = sorted(list(paths.list_images(base_dir + "happy/")))
    neutral_images = sorted(list(paths.list_images(base_dir + "neutral/")))
    sad_images = sorted(list(paths.list_images(base_dir + "sad/")))
    surprise_images = sorted(list(paths.list_images(base_dir + "surprise/")))

    images = []
    images.append(angry_images)
    images.append(disgust_images)
    images.append(fear_images)
    images.append(happy_images)
    images.append(neutral_images)
    images.append(sad_images)
    images.append(surprise_images)

    x_train = []
    y_train = []
    for k in images:
        for i in k:
            i = cv2.imread(i)
            i = img_to_array(i)
            x_train.append(i)
            y_train.append(images.index(k))


    x_train = np.array(x_train, dtype=np.float32) / 255.0
    y_train = np.array(y_train, dtype=np.int8)



    base_dir = "D:/Machine learning projects/Datasets/Face Expression/images/validation/"
    angry_images = sorted(list(paths.list_images(base_dir + "angry/")))
    disgust_images = sorted(list(paths.list_images(base_dir + "disgust/")))
    fear_images = sorted(list(paths.list_images(base_dir + "fear/")))
    happy_images = sorted(list(paths.list_images(base_dir + "happy/")))
    neutral_images = sorted(list(paths.list_images(base_dir + "neutral/")))
    sad_images = sorted(list(paths.list_images(base_dir + "sad/")))
    surprise_images = sorted(list(paths.list_images(base_dir + "surprise/")))

    images = []
    images.append(angry_images)
    images.append(disgust_images)
    images.append(fear_images)
    images.append(happy_images)
    images.append(neutral_images)
    images.append(sad_images)
    images.append(surprise_images)

    x_val = []
    y_val = []
    for k in images:
        for i in k:
            i = cv2.imread(i)
            i = img_to_array(i)
            x_val.append(i)
            y_val.append(images.index(k))


    x_val = np.array(x_val, dtype=np.float32) / 255.0
    y_val = np.array(y_val, dtype=np.int8)
    return (x_train,y_train,x_val,y_val)


import_data()