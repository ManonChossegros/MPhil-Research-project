
from mimetypes import init
import re
import numpy as np
import tifffile
from keras.callbacks import LearningRateScheduler
from numpy.random import seed
# seed(1)
import tensorflow as tf
from PIL import Image
from sys import getsizeof
from sklearn.utils import class_weight
from keras import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Convolution1D, Convolution2D, Dropout, MaxPooling2D, BatchNormalization, MaxPooling3D, Convolution3D, AveragePooling2D, Concatenate, AveragePooling3D, Reshape
from tensorflow.keras.optimizers import SGD
import os
from keras.callbacks import History
from tensorflow.python.client import device_lib
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.kerasExtendImageDataGenerator.keras.preprocessing.image2 import ImageDataGenerator

from tensorflow.keras.layers.experimental.preprocessing import Resizing

gpus = tf.config.list_logical_devices('GPU')

print(gpus)

def scheduler(epoch, lr):
  if epoch < 5:
    return lr_rate
  else:
    return lr * tf.math.exp(-0.1)

lr_rate = LearningRateScheduler(scheduler)
learning_rate = 00.0000001
loss_history = History()
callbacks_list = [loss_history, lr_rate]
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))



w = 2192
band = 10



def customized_image_reader(filepath, target_mode=None, target_size=None, dim_ordering='tf', **kwargs):
    imgTiff = tifffile.imread(filepath)

    return imgTiff




init_mode = 'uniform'
def create_model():
    input_shape = Input(shape = (1350, 500, 10))
    # input_shape = Reshape((1350,500, 19,1))(input_shape0)
    tower_1 = Convolution2D(20, (3,3), activation='relu', padding='same', kernel_initializer=init_mode)(input_shape)
    tower_1 = Convolution2D(8, (3, 3), activation='relu', padding = 'same', kernel_initializer = init_mode)(tower_1)
    tower_1 = MaxPooling2D((3,3))(tower_1)
    tower_1 = Dropout((0.3))(tower_1)
    
    BatchNormalization()
    tower_1 = Convolution2D(32, (5,5), padding='same')(tower_1)

    tower_2 = Convolution2D(32, (1,1), padding='same', kernel_initializer=init_mode)(input_shape)
    tower_2 = MaxPooling2D((3,3))(tower_2)
    tower_2 = Dropout((0.3))(tower_2)
    BatchNormalization()
    tower_2 = Convolution2D(32, (3,3), padding='same', kernel_initializer=init_mode)(tower_2)
    BatchNormalization()

    tower_3 = Convolution1D(filters=32, kernel_size=3, kernel_initializer=init_mode)(input_shape)
    BatchNormalization()
    tower_3 = MaxPooling2D((3,3))(tower_3)
    tower_3 = Dropout((0.3))(tower_3)
    merged = Concatenate(axis = 3)([tower_1, tower_2, tower_3])


    average_layer = AveragePooling2D((3,3))(merged)
    average_layer=Flatten()(average_layer)
    dense_layer = Dense(80, activation='relu')(average_layer)
    out = Dense(6, activation='sigmoid')(dense_layer)

    model = Model(input_shape, out)
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam', metrics = ['accuracy'])
    return model
def combine_generator(gen1, gen2):
    for data1, labels1 in gen1:
        for data2, labels2 in gen2: 
            yield np.concatenate((data1,data2)), np.concatenate((labels1,labels2))


datagen1 = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
datagen2 = ImageDataGenerator(rescale=1./255)

with tf.device('/gpu:0'):

    nb_epoch=6
    batchsize=20
    
    
    # train_it = datagen1.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=20, shuffle=True, target_size = (1350,500,10))
    # valid_it = datagen1.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Val', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=20, shuffle=True, target_size = (1350,500,10))
    train_it_1 = datagen1.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=5, shuffle=True, target_size = (1350,500,10))
    valid_it_1 = datagen1.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Val', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=5, shuffle=True, target_size = (1350,500,10))

    train_it_2 = datagen2.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=15, shuffle=True, target_size = (1350,500,10))
    valid_it_2 = datagen2.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Val', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=15, shuffle=True, target_size = (1350,500,10))
    train_it = combine_generator(train_it_1, train_it_2)
    valid_it = combine_generator(valid_it_1, valid_it_2)
    classes = []
    directory = '../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train'
    ### define classes
    nb_samples = 0
    for subdir in sorted(os.listdir('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train')):
        # print(subdir)
        if subdir != 0:
            if os.path.isdir(os.path.join('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train/', subdir)):
                classes.append(subdir)
   

    class_indices = dict(zip(classes, range(len(classes))))
    for subdir in classes:
        if subdir !=0:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                nb_samples += 1
    # print(nb_samples)
    classes = np.zeros((nb_samples,), dtype='int32')
    i = 0
    for subdir in classes:
        if subdir != 0:
            subpath = os.path.join(directory, subdir)
            for fname in os.listdir(subpath):
                classes[i] = class_indices[subdir]
                i+=1

    
    class_weights = class_weight.compute_class_weight(
           class_weight = 'balanced',
            classes = np.unique(classes), 
            y = classes)


    batchX, batchy = next(train_it)

    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchy.shape, batchX.min(), batchX.max()))
    model = create_model()
    print('bonjour')
    # model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam', metrics = ['accuracy'])
    model.fit(train_it, epochs=nb_epoch, steps_per_epoch=1250//20,
                    validation_data=valid_it, validation_steps = 287//20, class_weight = class_weights, callbacks = callbacks_list)

    model.save('model_Inception_ResNet_2_9h21_Fri_Augmented_quarter.h5')
