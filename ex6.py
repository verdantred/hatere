# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:15:21 2017

@author: wwnii

SGN-41007 Pattern Recognition and Machine Learning
Exercise Set 6
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from sklearn.datasets import load_files
from skimage.feature import local_binary_pattern
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Convolution2D, MaxPooling2D, Flatten, Dense


# 3

def normall(array):
    x_min = np.min(array)
    x_max = np.max(array) - x_min
    n_array = [ [ normali(ch, x_min, x_max) for ch in i] for i in images]
    return n_array
    
def normali(array, amin, amax):
    array -= amin
    array = np.true_divide(array, float(amax))
    return array

data = load_files('./GTSRB_subset_2', load_content=False)
images = np.array([image.imread(filename, "JPG") for filename in data.filenames])
images = np.array([np.transpose(i) for i in images])
images = normall(images)

X_train, X_test, y_train, y_test = train_test_split(images, data.target, test_size=0.2, random_state=58002)

# 4
N = 10      # Number of feature maps
w, h = 3, 3 # Conv. window size

model = Sequential()
model.add(Convolution2D(nb_filter = N,
                        nb_col = w,
                        nb_row = h,
                        activation = 'relu',
                        border_mode = 'same',
                        input_shape = (3,64,64)))

model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(nb_filter = N,
                        nb_col = w,
                        nb_row = h,
                        border_mode = 'same',
                        activation = 'relu'))

model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(2, activation = 'sigmoid'))
