
from gc import callbacks
import numpy as np
import tensorflow as tf
import tifffile
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Convolution2D, Dropout, MaxPooling2D, BatchNormalization, MaxPooling3D, Convolution3D
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
import os
from sklearn.utils import class_weight
from tensorflow.python.client import device_lib
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.kerasExtendImageDataGenerator.keras.preprocessing.image2 import ImageDataGenerator, standardize, random_transform
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(device_count={'GPU': 1}))

# tf.debugging.set_log_device_placement (True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

lr_rate = 0.0000001


def customized_image_reader(filepath, target_mode=None, target_size=None, dim_ordering='tf', **kwargs):
    imgTiff = tifffile.imread(filepath)
    return imgTiff
w = 2192
band = 10

lambda_list = [375, 405, 435, 450, 470, 505, 525, 570, 590, 630, 645, 660, 700, 780, 850, 870, 890, 940, 970]
# print(len(lambda_list))

dic_wave_length = {}
for k in range(band):
    dic_wave_length[k] = lambda_list[k] 


def save_variable(variable, txt_file):
    # variables = [X_train_all, Y_train_all, cube_healthy_1, cube_YR_1, cube_YR_2, cube_YR_3]
    Savingvariables = open(txt_file,"wb")
    pickle.dump(variable, Savingvariables)
    Savingvariables.close()

def collect_variable(txt_files, filenames=None, target_mode=None, target_size=None, dim_ordering=None, class_mode='categorical', classes=None, directory=None, nb_sample=None,seed=None, sync_seed= None):
        # if I want to take them back
    fichierini = txt_files
    fichierSauvegarde = open(fichierini,"rb")
    variable = pickle.load(fichierSauvegarde) 
    variable = np.reshape(variable, (1350,500,10,1))/255
    return variable
    
print(device_lib.list_local_devices())   


def scheduler(epoch, lr):
  if epoch < 5:
    return lr_rate
  else:
    return lr * tf.math.exp(-0.1)

def combine_generator(gen1, gen2):
    for data1, labels1 in gen1:
        for data2, labels2 in gen2: 
            yield np.concatenate((data1,data2)), np.concatenate((labels1,labels2))


callback = tf.keras.callbacks.LearningRateScheduler(scheduler) 


def create_model():

    model = Sequential()
    model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(1350, 500, 10), padding = 'same'))
    BatchNormalization()
    model.add(Convolution2D(36, 3, 3, activation='relu', padding = 'same'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(8, 3, 3, activation='relu', padding = 'same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Dropout(0.3))
    BatchNormalization()

    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    BatchNormalization()


    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation = 'sigmoid'))

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])



    return model

datagen1 = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
datagen2 = ImageDataGenerator(rescale=1./255)
with tf.device('/gpu:0'):
    nb_epoch=6
    batchsize=20

    
    # train_it = datagen.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=5, shuffle=True, target_size = (1350,500,10))
    # print(len(train_it.classes))
    train_it_1 = datagen1.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=5, shuffle=True, target_size = (1350,500,10))
    valid_it_1 = datagen1.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Val', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=5, shuffle=True, target_size = (1350,500,10))
    train_it_2 = datagen2.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=15, shuffle=True, target_size = (1350,500,10))
    valid_it_2 = datagen2.flow_from_directory('../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Val', image_reader=customized_image_reader, read_formats={'tiff'}, class_mode='categorical', batch_size=15, shuffle=True, target_size = (1350,500,10))
    train_it = combine_generator(train_it_1, train_it_2)
    valid_it = combine_generator(valid_it_1, valid_it_2)

    ### define class weights
    classes = []
    directory = '../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Train'
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
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    ## fit model
    model = create_model()
    history = model.fit_generator(train_it, epochs=nb_epoch, steps_per_epoch=1013//20,
                    validation_data=valid_it, validation_steps=287//20, callbacks = [callback])

    
    model.save('model_2DCNN_augmented_with_class_wieghts_17h29_Thurs.h5')



