
# proccessing image 
from keras.preprocessing.image import ImageDataGenerator , load_img
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# xx = 'apa.pp'
# print(xx.split('.')[0])


# if data in the same folder categories it in the data frame
def is_A_B(filename):
    file_name = os.listdir(filename)
    categories  = []
    for file in file_name:  #for each file in the folder
        category = file.split('.')[0] #to get the first word in front of (.)
        # print(categore)
        if category == 'cats':
            categories .append(0) #every file has name cat put it in list = 0 else =1
        else:
            categories .append(1)
    return categories
        
# create data frame to contain name cate = 0 else =1
# data = pd.DataFrame({
#         'filename':file_name,
#         'categories':categories
#     })


train_datagen = ImageDataGenerator(rotation_range=15,
                                    rescale=1./255,
                                    shear_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale = 1./255)
valid_dategen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (128, 128),
                                            batch_size = 64,
                                            class_mode = 'binary')
valid_set =  valid_dategen.flow_from_directory('test_set',
                                            target_size = (128, 128),
                                            batch_size = 64,
                                            class_mode = 'binary')
# show and plot some figers from data set

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in training_set:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()



# calling keras library and pakage
import keras
from keras.models import Sequential
from keras.layers import Dense , Flatten , Conv2D 
from keras.layers import  MaxPooling2D ,BatchNormalization , Dropout
# install model
classifier = Sequential()

# 1- convolution
classifier.add(Conv2D(32,3,input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(BatchNormalization())
# 2- pooling image
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

#3-  Adding a second convolutional layer
classifier.add(Conv2D(32, 3, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25)) #used for prevent overfitting model
#3-  Adding a third convolutional layer
classifier.add(Conv2D(64, 3, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection and hidden layers
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(BatchNormalization())
# classifier.add(Dense(units = 64, activation = 'relu'))
# classifier.add(BatchNormalization())
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

# callbacks if stop the train model if acc -100 or loss > 1
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

#valid_steps  = (validTotal // batchsize)  ,, steps per epoch = (TreainTotal // batchsize)

# fitting data and train
classifier.fit_generator(training_set,
                         steps_per_epoch = 125,
                         epochs = 25,
                         callbacks= callbacks,
                         validation_data = valid_set,
                         validation_steps= 3
                         )


# save cnn model
classifier.save('cat-dog-dedection.h5')

# load cnn modek
model = keras.models.load_model('cat-dog-dedection.h5')

# image proccessing for one image to predict if it A or B
from skimage import transform
from PIL import Image as image
img = 'Cat.jpg'
img = image.open(img)  #read and of iamge
np_img = np.array(img).astype('float32')/255  #convert iamge to array 
np_img = transform.resize(np_img , (128,128,3)) # resize iamge and proccessing it 
np_img = np.expand_dims(np_img, axis=0)
# img.show()
y_pred = model.predict(np_img)
if y_pred > 0 and y_pred > 0.5 :
    y_pred = 'cat'
else:
    y_pred = 'dog'
# idx_imagen = np.random.randint(0, n_letters_heldOut)
plt.figure(figsize = (6,6))
# plt.title("It's character "+ str(y_held[idx_imagen,0]) + "  /  It's tagged then " + str(y_pred[idx_imagen]))
plt.title(y_pred)
# img=X_held[idx_imagen, :, :]

plt.imshow(img)
plt.show()