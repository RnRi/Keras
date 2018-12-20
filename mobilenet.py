import os
import tensorflow as tf
from keras.models import Sequential, Model
from keras.models import load_model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from tensorflow.python.framework import graph_util

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

K.set_image_dim_ordering('tf')


MN2 = MobileNetV2(weights='imagenet', include_top=True)
for layer in MN2.layers:
    layer.trainable = False

x = Dense(15, activation='softmax')(MN2.layers[-2].output)

model = Model(input=MN2.input, output=x)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 100

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '/data/image_net/train/train',  # this is the target directory
        target_size=(224, 224),  
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '/data/image_net/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

model.save('model.h5')


