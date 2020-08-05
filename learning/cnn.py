import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
import random
import h5py
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model


def learning():

    epoch = 30
    cnn = build_model(100,100,30)
    X_tr, X_val = preprocess_dataset()

    cnn.fit(X_tr, steps_per_epoch=60955//32, epochs=epoch, validation_data=X_val, validation_steps=6737//32)
    cnn.save('model.h5')
    

def predict_test_set():

    cnn = tf.keras.models.load_model('model.h5')
    test_gen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1)
    train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, validation_split=0.1)
    train = train_gen.flow_from_directory('fruits-360/Training', target_size=(100, 100), shuffle=True, seed=13, batch_size=32, class_mode='categorical', subset='training', color_mode='rgb')
    test = test_gen.flow_from_directory('fruits-360/Test', target_size=(100, 100), shuffle=False, seed=13, class_mode=None, batch_size=1, color_mode='rgb')
    test.reset()

    pred = cnn.predict(test, verbose = 1, steps= 22688)
    pred_indices = np.argmax(pred, axis=1)
    label = (train.class_indices)
    label = dict((v,k) for k,v in label.items())
    predicted_list = [label[k] for k in pred_indices]
    filenames = test.filenames
    results = pd.DataFrame({'Filename': filenames, 'Prediction': predicted_list})

    print(results)

    
def preprocess_dataset():

    train_gen = ImageDataGenerator(
        rescale=1./255, shear_range=0.1, zoom_range=0.1, validation_split=0.1)
    test_gen = ImageDataGenerator(
        rescale=1./255, shear_range=0.1, zoom_range=0.1)

    training = train_gen.flow_from_directory('fruits-360/Training', target_size=(
        100, 100), shuffle=True, seed=13, batch_size=32, class_mode='categorical', subset='training', color_mode='rgb')
    validation = train_gen.flow_from_directory('fruits-360/Training', target_size=(
        100, 100), shuffle=True, seed=13, batch_size=32, class_mode='categorical', subset='validation', color_mode='rgb')
    test = test_gen.flow_from_directory('fruits-360/Test', target_size=(
        100, 100), shuffle=False, seed=13, class_mode=None, batch_size=1, color_mode='rgb')

    return training, validation


def build_model(width, heigth, epochs):

    input_shape = (width, heigth, 3)
    opt = Adam(lr = 1e-3, decay= 1e-3/epochs)

    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(131))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


#learning()
predict_test_set()
