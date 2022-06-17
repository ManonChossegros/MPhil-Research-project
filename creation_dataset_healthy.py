import tifffile
from cv2 import log
import rawpy
import imageio
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
from PIL import  Image, ImageColor, ImageDraw, ImageEnhance
from spectral import *
import spectral.io.aviris as aviris
import matplotlib
# matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
# import wx
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from scipy.signal import argrelextrema, correlate2d
from scipy.interpolate import interp1d
import scipy
import cv2
from sklearn import svm
import pickle
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random
import cv2
spectral.settings.WX_GL_DEPTH_SIZE = 16
from skimage.restoration import estimate_sigma
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Convolution2D, Dropout, MaxPooling2D, BatchNormalization, MaxPooling3D, Convolution3D
from tensorflow.keras.optimizers import SGD
from keras.initializers import random_uniform
import tensorflow as tf
from sklearn.decomposition import PCA 
tf.config.run_functions_eagerly(True)

with open('log_creation_dataset.txt', 'w') as logfile: 
    logfile.write('import of all packages')
    logfile.write('\n')


w = 2192
band = 19

lambda_list = [375, 405, 435, 450, 470, 505, 525, 570, 590, 630, 645, 660, 700, 780, 850, 870, 890, 940, 970]





'''''Creation of the cube'''

def create_raw_cube(path):
    cube = np.zeros((2192, 2192, 19))


    for j in range(1, 20): 
        if j <10:
            im =  Image.open(path + '_0' + str(j) + '.png').convert("L")
        else: 
            im =  Image.open(path + '_' + str(j) + '.png').convert("L")
           
        slice = np.array(im.getdata())
        slice = np.reshape(slice, (w,w))

        cube[:,:,j-1] = slice
    return cube


def background_removal(path, raw_cube_of_interest):
    img= cv2.imread(path + '_15.png')
    imgCont=img.copy()

    ## Convert to Gray
    imgGray =255- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Show the different
    blur = cv2.GaussianBlur(imgGray,(5,5),0)
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(blur, kernel, iterations=4)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=4)
    # imshow(img_erosion)

    closing = cv2.morphologyEx(255-img_erosion,cv2.MORPH_CLOSE,kernel, iterations = 3)
    # imshow(closing)
    thresh = np.where(closing >80, 255, 0)
    threshcopy = np.uint8(thresh)
    
    ## Find edges
    # imgEdges = cv2.Canny(closing,40,90)
    # imshow(imgEdges)
    # thresh = cv2.threshold(imgEdges, 128, 255, cv2.THRESH_BINARY)[1]
    img0 = np.zeros((w,w,3))
    contours,hierarchy =cv2.findContours(threshcopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    
    ## Loop through contours and find the two biggest area.
    contour_list = []
    for cont in contours:
        area=cv2.contourArea(cont)

        if area > 10000:
            # print(area)
            contour_list.append(cont)
    cv2.drawContours(img0,contour_list,-1,(255,0,0), thickness=cv2.FILLED)
    mask = np.where(img0[:,:,0] !=0, 1, 0)
    cropped_cube = np.zeros((w,w,band))
    for k in range(band):
        cropped_cube[:,:,k] = np.multiply(raw_cube_of_interest[:,:,k], mask)
    return cropped_cube, mask

def find_left_right_border(background_mask):
    height_mask = np.shape(background_mask)[0]
    width_mask = np.shape(background_mask)[1]
    counter_begin = 0
    leaf_begin = 0
    leaf_end = 0    
    for k in range(width_mask): 
        if background_mask[height_mask//2, k] == 255: 
            counter_begin+=1
            if counter_begin >30:
                leaf_begin = k-counter_begin
                break
        if background_mask[height_mask//2, k ]==0:
            counter_begin = 0
    counter_end = 0
    for k in range(width_mask-1, 0, -1):
        if background_mask[height_mask//2, k] == 255: 
            counter_end+=1
            if counter_end >30:
                leaf_end = k+counter_end
                break
        if background_mask[height_mask//2, k]==0:
            counter_end = 0     
    return leaf_begin - 100, leaf_end+100

def find_window(background_mask):
    left_border, right_border = find_left_right_border(background_mask)
    top_border = 450
    bottom_border = 1800

    return  top_border, bottom_border, left_border, right_border


def crop_and_background_removal(path, raw_cube_of_interest):
    img= cv2.imread(path + '_15.png')
    imgCont=img.copy()

    ## Convert to Gray
    imgGray =255- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Show the different
    blur = cv2.GaussianBlur(imgGray,(5,5),0)
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(blur, kernel, iterations=4)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=4)
    # imshow(img_erosion)

    closing = cv2.morphologyEx(255-img_erosion,cv2.MORPH_CLOSE,kernel, iterations = 3)
    # imshow(closing)
    thresh = np.where(closing >80, 255, 0)
    top_border, bottom_border, left_border, right_border = find_window(thresh)  
    print(top_border, bottom_border, left_border, right_border) 
    cropped_mask = thresh[top_border:bottom_border, left_border:right_border]
    cropped_cube = raw_cube_of_interest[top_border:bottom_border, left_border:right_border,:]
    threshcopy = np.uint8(cropped_mask)
    dim = np.shape(cropped_cube)

    img0 = np.zeros((dim[0],dim[1],3))
    contours,hierarchy =cv2.findContours(threshcopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    
    ## Loop through contours and find the two biggest area.
    contour_list = []
    for cont in contours:
        area=cv2.contourArea(cont)

        if area > 10000:
            # print(area)
            contour_list.append(cont)
    # imshow(img0)
    cv2.drawContours(img0,contour_list,-1,(255,0,0), thickness=cv2.FILLED)
    mask = np.where(img0[:,:,0] !=0, 1, 0)
    
    bg_removed = np.zeros((dim[0],dim[1],band))
    for k in range(band):
        bg_removed[:,:,k] = np.multiply(cropped_cube[:,:,k], mask)
    return bg_removed



def create_smooth_cube(cube_of_interest):
    dim = np.shape(cube_of_interest)
    smoothed_cube = np.zeros((dim[0], dim[1], dim[2]), dtype=np.uint8)

    smoothed_cube[:,:,:] = scipy.signal.savgol_filter(cube_of_interest[:,:,:], axis = 2, deriv = 0, polyorder = 3, window_length= 11, delta=1)
    return smoothed_cube

def create_final_cube (path):
    cube = create_raw_cube(path)
    smooth_cube = create_smooth_cube(cube)
    final_cube = crop_and_background_removal(path, smooth_cube)
    return final_cube


def extract_leaf(final_cube):
    img = np.uint8(final_cube[:,:,15])
    dim = np.shape(img)
    contours,hierarchy =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    
    ## Loop through contours and find the largest areas
    contour_list = []
    leaf_list = []
    for cont in contours:
        contour_list = []
        img0 = np.zeros((dim[0],dim[1],3))
        area=cv2.contourArea(cont)
        if area > 10000:
            contour_list.append(cont)
            # print(np.shape(contour_list))
            cv2.drawContours(img0,contour_list,-1,(255,0,0), thickness=cv2.FILLED)
            mask = np.where(img0[:,:,0] !=0, 1, 0)
        ## apply the mask to the image
            img_without_background = np.zeros((dim[0],dim[1],band))
            for k in range(band):
                img_without_background[:,:,k] = np.multiply(final_cube[:,:,k], mask)
            left_border_0, right_border = find_left_right_border(mask*255)
            left_border = np.max([0, left_border_0])
            
            if left_border + 500 < dim[1]-1:
                cropped_img_without_background = img_without_background[:, left_border: left_border+500,:]
            else: 
                cropped_img_without_background = img_without_background[:, -500:,:]
            leaf_list.append(cropped_img_without_background)
    return leaf_list



index_cube = 0

for k in range(1, 92):
    index_cube +=1
    print(index_cube)
    final_cube_healthy = create_final_cube('../../projects/niab/mchosseg/Camera_2/Healthy/png_images/Healthy_29_March_Vuka_Rep' + str(k))
    # dic_leaves_healthy[index_cube] = extract_leaf(final_cube_healthy)
    leaf_list = extract_leaf(final_cube_healthy)
    index_leaf = 0
    for j in leaf_list:
        index_leaf+=1
        tifffile.imsave( '../../projects/niab/mchosseg/Camera_2/Cubes/Reduced/Healthy Rep' + str(k) +' leaf ' + str(index_leaf) + " reduced.tiff", j[:,:,[7,8,10,11,12,13,15,16,17,18]])

