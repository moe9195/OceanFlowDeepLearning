'''
Model setup and training
'''

# importing relevant packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import pickle, csv, fileinput, shutil, os, re
import warnings
from ocean import *

# exception handling to suppress errors
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Error class to handle errors
class Error(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

# normalize data to [0, 1]
def normalize(x, min_val, max_val):
    return (x - min_val)/(max_val - min_val)

def normalize_batch(x):
    return (x-x.min())/(x.max() - x.min())

# split square matrix into 4 equal sized matrices
def split_in_four(A):
        shape = np.shape(A)
        d, l, w = shape[0], shape[1], shape[2]
        div = int(l/2)
        upperLeft  = A[0:d,0:div,0:div]
        upperRight = A[0:d,0:div,div:2*div]
        lowerLeft  = A[0:d,div:2*div,0:div]
        lowerRight = A[0:d,div:2*div,div:2*div]
        combined   = np.concatenate((upperLeft, upperRight, lowerLeft, lowerRight))
        return combined

 generator that outputs batches of 60x60 array data from a directory or data.
# variable is either 'temperature' or 'kinetic energy'
# subset is 'train', 'validate' or 'test'
# scale = True to normalize data, otherwise, data isn't normalized
# split = True to split each 2D array into 4 smaller arrays
def generator(files, variable, subset, scale_type, split, batch_size = 64):
    if split:
        batch_size = batch_size//4

    while True:
        batch_files = np.random.choice(a=files, size=batch_size)
        batch_X = np.empty((batch_size, 60, 60, 1))
        batch_Y = np.empty((batch_size, 60, 60, 1))

        for idx, file in enumerate(batch_files):
            X_temp = np.load('./dataset/'+variable+'/' + subset + '/X/'+file)
            Y_temp = np.load('./dataset/'+variable+'/' + subset + '/Y/'+file)
            if scale_type == 'single':
                X_temp = normalize_batch(X_temp)
                Y_temp = normalize_batch(Y_temp)

            batch_X[idx] = np.expand_dims(X_temp, 3)
            batch_Y[idx] = np.expand_dims(Y_temp, 3)

        # normalize data to [0, 1] based on global maximum and minimum values
        # for each variable
        if scale_type == 'full':
            if variable == 'kinetic energy':
                batch_X = (batch_X - -0.0)/(6.6047177 - -0.0)
                batch_Y = (batch_Y - -24.878328)/(19.028616 - -24.878328)
            if variable == 'temperature':
                batch_X = (batch_X - -2.7502365)/(32.541855 - -2.7502365)
                batch_Y = (batch_Y - -24.878328)/(19.028616 - -24.878328)
        elif scale_type == 'batch':
            batch_X = normalize_batch(batch_X)
            batch_Y = normalize_batch(batch_Y)

        if split:
            batch_X = split_in_four(batch_X)
            batch_Y = split_in_four(batch_Y)
        yield (batch_X, batch_Y)

# combines the data into an NxMxMx1 array of data
# this can be used to train the network instead of using a generator
# however, it requires lots of memory to store the array (~30GB)
def combine(files, subset):
    T = np.empty((len(files), 60, 60, 1))
    KE = np.empty((len(files), 60, 60, 1))
    VOR = np.empty((len(files), 60, 60, 1))
    for idx, file in enumerate(files):
        T_temp = np.load('./dataset/temperature/' + subset + '/X/'+file)
        KE_temp = np.load('./dataset/kinetic energy/' + subset + '/X/'+file)
        VOR_temp = np.load('./dataset/temperature/' + subset + '/Y/'+file)
        T[idx] = T_temp.reshape((60, 60, 1))
        KE[idx] = KE_temp.reshape((60, 60, 1))
        VOR[idx] = VOR_temp.reshape((60, 60, 1))
    return T, KE, VOR


# define model hyperparameters
batch_size = 64
num_epoch = 100
imsize = 60

# define the directories for each subset
path_train = './dataset/kinetic energy/train/'
path_validate = './dataset/kinetic energy/validate/'
path_test = './dataset/temperatkinetic energyure/test/'
files_train = os.listdir(path_train + 'X')
files_validate = os.listdir(path_validate + 'X')
files_test = os.listdir(path_test + 'X')

# create generators for training, validation and testing data
train_gen = generator(files_train, variable='kinetic energy',
                      subset='train', scale_type = 'full',
                      split = False, batch_size = batch_size)
val_gen   = generator(files_validate, variable='kinetic energy',
                      subset='validate', scale_type = 'full',
                      split = False, batch_size = batch_size)
test_gen = generator(files_validate, variable='kinetic energy',
                      subset='validate', scale_type = 'full',
                      split = False, batch_size = batch_size)


# building the model architecture and compiling the model

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(imsize, imsize, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(32, (3, 3), padding='same', dilation_rate=1))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64, (3, 3), padding='same', dilation_rate=2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128, (3, 3), padding='same', dilation_rate=4))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(256, (3, 3),  padding='same', dilation_rate=4))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dense(1))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam())

model.summary()


# save best weights
checkpoint = ModelCheckpoint(filepath='best_weights_KE_full.hdf5',
                             save_best_only=True, save_weights_only=True)

# reduce learning rate if learning doesn't improve for two consecutive epochs
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.75,
                              patience=5, verbose=2, mode='max')

# model training
hist = model.fit_generator(
           train_gen, steps_per_epoch=len(files_train) // batch_size,
           epochs=num_epoch, validation_data=val_gen,
           validation_steps=len(files_validate) // batch_size, callbacks=[lr_reduce, checkpoint])


# saving model and training history
model.save('my_model_KE_full.h5')

history_dict = hist.history
f = open('history_KE_full.pckl', 'wb')
pickle.dump(history_dict, f)
f.close()
