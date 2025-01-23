

import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
y_train = to_categorical(y_train , 10) #data , num of classes
y_test=to_categorical(y_test,10)

x_test = x_test/255
x_train = x_train/255

import keras
from keras.models import Sequential
from keras.layers import Dense , Flatten , Conv2D 
from keras.layers import  MaxPooling2D ,BatchNormalization , Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas

# install model
classifier = Sequential()

# 1- convolution
classifier.add(Conv2D(32,3,input_shape = (32, 32, 3), activation = 'relu', padding='same'))
classifier.add(BatchNormalization())
# 2- pooling image
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

#3-  Adding a second convolutional layer
classifier.add(Conv2D(64, 3, activation = 'relu' , padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#3-  Adding a third convolutional layer
classifier.add(Conv2D(64, 3, activation = 'relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.3))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection and hidden layers
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# The code contains an error in the last layer of the model. You are trying to predict 10 classes in CIFAR-10 dataset, so the last layer should have 10 units instead of 1. Replace units = 1 with units = 10 in the last layer.

classifier.summary()

# callbacks if stop the train model if acc -100 or loss > 1
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop]

#valid_steps  = (validTotal // batchsize)  ,, steps per epoch = (TreainTotal // batchsize)

# fitting data and train

classifier.fit(x_train
                         ,y_train,
                         epochs = 25,
                         callbacks= callbacks,
                         validation_data = (x_test,y_test),
                         validation_steps= 3
                         )

