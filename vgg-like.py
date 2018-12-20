import tensorflow as tf
import numpy as np
import sys
import cv2
import os
from keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
print(tf.VERSION)
print(tf.keras.__version__)

# os.environ["CUDA_VISIBLE_DEVICES"]= "7" 


datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input,
                             horizontal_flip=True)
datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = datagen_train.flow_from_directory('/data/image_net/train/train', target_size=(224, 224), batch_size=64)
val_generator = datagen_test.flow_from_directory('/data/image_net/validation', target_size=(224, 224), batch_size=64)
#test_generator = datagen_test.flow_from_directory('/data/image_net/test', target_size=(224, 224), batch_size=64)


#create model
def build_model(drop=True):

    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))

    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, strides=2))

    model.add(Flatten())
    if drop:
        model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    if drop:
        model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))

    # model.add(ZeroPadding2D((1, 1), input_shape=x_train.shape[1:]))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(128, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(128, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(Flatten())
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))
    return model

if sys.argv[1] == 'train':
    model = build_model()
    tf.contrib.quantize.create_training_graph()
    saver = tf.train.Saver()
    sess = tf.Session()

    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit_generator(train_generator, validation_data=val_generator,
                        epochs=25, workers=8, callbacks=[tf.keras.callbacks.TensorBoard('./eval_deep/train')])
    

    # preds = model.predict_generator(test_generator)
    # print(confusion_matrix(test_generator.classes, np.argmax(preds, axis=-1)))
    # print(classification_report(test_generator.classes, np.argmax(preds, axis=-1)))

    saver.save(sess, './eval/model.ckpt')
    model.save('./eval/model.h5')



else:
    model = build_model(drop=False)
    tf.contrib.quantize.create_eval_graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './eval/model.ckpt')
        train_writer = tf.summary.FileWriter('./eval/test', sess.graph)
        saver.save(sess, './eval/model_eval.ckpt')

