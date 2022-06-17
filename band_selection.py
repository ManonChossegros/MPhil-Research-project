import rawpy
import imageio
import numpy as np
from numpy.random import seed
import matplotlib
seed(1)
matplotlib.rc('font',family='Times New Roman')
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tensorflow import random
random.set_seed(2)
from PIL import  Image, ImageColor, ImageDraw, ImageEnhance
from spectral import *
import xml.etree.ElementTree as ET
import spectral.io.aviris as aviris
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import wx
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from scipy.signal import argrelextrema, correlate2d
from scipy.interpolate import interp1d
import scipy
import math
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
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2 
import tensorflow as tf
from sklearn.decomposition import PCA
import os 
from scipy.signal import savgol_filter
tf.config.run_functions_eagerly(True)
from scipy.interpolate import interp1d



w = 2192
band = 19

lambda_list = [375, 405, 435, 450, 470, 505, 525, 570, 590, 630, 645, 660, 700, 780, 850, 870, 890, 940, 970]
print(len(lambda_list))

dic_wave_length = {}
for k in range(band):
    dic_wave_length[k] = lambda_list[k] 

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

def background_removal_0(cube_of_interest):
    h = int((np.shape(cube_of_interest)[0]))
    cropped_cube = np.zeros((h,w,band))
    # vi = np.where(b==0, 0, a/b) 
    index_1= cube_of_interest[:,:,16]
    
    back_ground_mask_0 = np.where(index_1 >100, 1, 0)

    

    for k in range(band):
        cropped_cube[:,:,k] = np.multiply(cube_of_interest[:,:,k],back_ground_mask_0)
    
    cropped_cube_2 = np.zeros((h,w,band))
  
    index_2 = cropped_cube[:,:,18]- 2*cropped_cube[:,:,0]
    back_ground_mask_1 = np.where(index_2>60, 255,0 )
    for k in range(band):
        cropped_cube_2[:,:,k] = np.multiply(cube_of_interest[:,:,k],back_ground_mask_1)
    imshow(back_ground_mask_1)

    return cropped_cube_2, back_ground_mask_1

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
    imshow(closing)
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
    imshow(img0)
    cv2.drawContours(img0,contour_list,-1,(255,0,0), thickness=cv2.FILLED)
    mask = np.where(img0[:,:,0] !=0, 1, 0)
    cropped_cube = np.zeros((w,w,band))
    for k in range(band):
        cropped_cube[:,:,k] = np.multiply(raw_cube_of_interest[:,:,k], mask)
    return cropped_cube, mask

def find_left_right_border(background_mask):
    counter_begin = 0
    leaf_begin = 0
    leaf_end = 0    
    for k in range(w): 
        if background_mask[1096, k] == 255: 
            counter_begin+=1
            if counter_begin >30:
                leaf_begin = k-counter_begin
                break
        if background_mask[1096, k ]==0:
            counter_begin = 0
    counter_end = 0
    for k in range(w-1, 0, -1):
        if background_mask[1096, k] == 255: 
            counter_end+=1
            if counter_end >30:
                leaf_end = k+counter_end
                break
        if background_mask[1096, k]==0:
            counter_end = 0     
    return leaf_begin - 100, leaf_end+100

def find_window(cube_of_interest, background_mask):
    left_border, right_border = find_left_right_border(background_mask)
    print(left_border,right_border)
    index_1= cube_of_interest[:,left_border:right_border,1]
    # imshow(index_1)
    # back_ground_mask_0 = np.where(index_1 >50, 255, 0)
    back_ground_mask_0 = np.where(index_1 >100, 255, 0)
    # imshow(back_ground_mask_0)
    dim_mask_0 = np.shape(back_ground_mask_0)
    imgGray = np.reshape(back_ground_mask_0, (dim_mask_0[0], dim_mask_0[1], 1))
    im0 = np.zeros((dim_mask_0[0], dim_mask_0[1], 3))
    slice1Copy = np.uint8(imgGray) 

    contours,hierarchy =cv2.findContours(slice1Copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for cont in contours:
        area=cv2.contourArea(cont)
        cv2.drawContours(slice1Copy,cont,-1,(255,0,0),5)
        # print(area)
        # print(cont)
        if area > 200:
            # print(area)
            # cv2.drawContours(slice1Copy,cont,-1,(255,0,0),5)
            cv2.drawContours(im0,cont,-1,(255,0,0),5, )
            
        # cv2.fillPoly(img, cont, color=list_colour[j])
    mask_tack = np.where(im0[:,:,0] !=0, 1, 0)
    # imshow(slice1Copy)
    imshow(mask_tack)
    tack_pixels_top = np.nonzero(mask_tack[:1096,:])[0]
    tack_pixels_bottom = np.nonzero(mask_tack[1096:,:])[0]
    top_border = np.max(tack_pixels_top) + 20
    bottom_border = np.min(tack_pixels_bottom) - 20 + 1096
    

    # imgGray = np.reshape(back_ground_mask_0, (w, w, 1))
    # slice1Copy = np.uint8(imgGray)
    # img_colour = np.zeros((w,w,3))
    # idx = np.nonzero(slice1Copy)
    # img_colour[idx[0], idx[1],:] = [255,255, 255]

    ## Find contour
    # contours,hierarchy =cv2.findContours(slice1Copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

    return  top_border, bottom_border, left_border, right_border

def create_smooth_cube_Savitzky_Golay(pixel_spectrum):
    f1 = interp1d(lambda_list, pixel_spectrum)
    
    xnew = np.concatenate((np.linspace(lambda_list[0], lambda_list[-1], 300), lambda_list))
    xnew.sort()
    
    
    # print(xnew)
    list_index = [1, 17, 33, 41, 52, 71, 82, 105, 117, 138, 146, 155, 176, 217, 253, 264, 275, 301, 317]
    fnew1 = f1(xnew)
    
    smooth_curve_1 = savgol_filter(fnew1, 31, 3)
    # return smooth_curve_1[list_index]
    return smooth_curve_1

def moving_average(pixel):

    pixel_smoothed = [np.mean(pixel[:2])]
    for j in range(1, 18):
        pixel_smoothed.append(np.mean(pixel[j-1:j+2]))
    pixel_smoothed.append(np.mean(pixel[-2:]))

    return(pixel_smoothed)

def create_smooth_cube_moving_average(list_all_ROI):

    ##show the average spectrum of smoothed vs not smoothed leaves
    pixel0 = np.mean(list_all_ROI, axis=0)
    pixel1 = np.mean(pixel0, axis=0)
    pixel = np.mean(pixel1,axis=0)

    pixel_smoothed = moving_average(pixel)

    plt.figure()
    plt.plot(lambda_list, pixel, label = 'Not smoothed pixel')
    plt.plot(lambda_list, pixel_smoothed, label = 'Smoothed pixel by moving average')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Pixel Intensity')
    plt.title('Moving average on the spectrum of leaves infected by Septoria')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1))
    plt.show(block=False)

    ## smooth every ROI
    list_all_ROI_smoothed = np.apply_along_axis(moving_average, 3, np.array(list_all_ROI))
    return list_all_ROI_smoothed
    
def create_final_cube (path):
    cube = create_raw_cube(path)
    bg_removed_cube, background_mask = background_removal(path, cube)
    top_border, bottom_border, left_border, right_border = find_window(cube, background_mask*255)
    print('window cube: ', top_border, bottom_border, left_border, right_border)
    cropped_cube = bg_removed_cube[top_border:bottom_border, left_border:right_border,:]
    ### if I want to show
    # imshow(cube)
    # imshow(cropped_cube)
    # imshow(background_mask)
    return cropped_cube
def extract_ROI(final_cube, M=30,N=30):
    # final cube has no tack and no background
    im = np.uint8(final_cube[:,:,[10,15,17]])
   
    # im = np.uint8(np.reshape(im0, (np.shape(im0)[0],np.shape(im0)[1],1)))
    im2 = im.copy()
    # gray2 = np.uint8(np.reshape(im, (np.shape(im)[0],np.shape(im)[1],1)))

    tiles = [im2[x:x+M,y:y+N,2] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    index = [[x,x+M, y, y+N,2]for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
    # print(len(tiles), len(index))
    # print(index)
    index_non_zero=[]
    non_zeros_tiles = []
    for k in range(len(tiles)): 
        
        tile = tiles[k]
        if np.count_nonzero(tile) == M*N:
            
            index_non_zero = index[k]
            non_zeros_tiles.append(final_cube[index_non_zero[0]: index_non_zero[1],index_non_zero[2]: index_non_zero[3],:])
            # print(index_non_zero)
        
            
            im[int(index_non_zero[0]):int(index_non_zero[1]), int(index_non_zero[2]):int(index_non_zero[3])] = [255,255,255]
            im[index_non_zero[0]:index_non_zero[1], index_non_zero[2]] = [255,0,0]
            im[index_non_zero[0]:index_non_zero[1], index_non_zero[3]] = [255,0,0]
            im[index_non_zero[0], index_non_zero[2]: index_non_zero[3]] = [255,0,0]
            im[index_non_zero[1], index_non_zero[2]: index_non_zero[3]] = [255,0,0]
            # imshow(im3)

    imshow(im)
    return non_zeros_tiles


def show_each_wavelength_image(cube_of_interest):
    fig, axs = plt.subplots(4, 5)


    for j in range(0, 20): 
        # print(j//4, j%5)
        axs[j//5][j%5].imshow(cube_of_interest[:,:,j-1], cmap = 'jet')
        axs[j//5][j%5].set_title('lambda = ' + str(lambda_list[j-1]) + ' nm', fontsize = 5)
        axs[j//5][j%5].get_xaxis().set_visible(False)
        axs[j//5][j%5].get_yaxis().set_visible(False)
        # plt.title('lambda = ' + str(lambda_list[j-1]) + ' nm')
        # plt.colorbar(axs[j//5][j%4])
    plt.show(block = False)

def find_band_of_interest(list_box_1, list_box_2,plot = True):
    ttest_list = []
    pvalue_list = []
    list_box_1 = np.array(list_box_1)
    list_box_2= np.array(list_box_2)
    for k in range(band):
        ttest_list.append(scipy.stats.ttest_ind(list_box_1[:,k], list_box_2[:,k], nan_policy='omit')[0])
        pvalue_list.append(scipy.stats.ttest_ind(list_box_1[:,k], list_box_2[:,k], nan_policy='omit')[1])
        # ttest_list.append(scipy.stats.ttest_ind(list_box_1[:][k], list_box_2, nan_policy='omit')[0])
        # pvalue_list.append(scipy.stats.ttest_ind(list_box_1[:][k], list_box_2, nan_policy='omit')[1])


    if plot == True: 
        plt.figure()
        matplotlib.rc('font',family='Times New Roman')
        plt.plot(lambda_list, pvalue_list)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('pvalue')
        plt.title('pvalue of the ttest between healthy and mildew with yellow rust pixels', y = 1.04)
        plt.show(block=False)

        plt.figure()
        matplotlib.rc('font',family='Times New Roman')
        plt.plot(lambda_list, np.abs(ttest_list))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('t-value')
        plt.title('t-value of the ttest between healthy and mildew with yellow rust pixels', y=1.04)
        # plt.scatter([631, lambda_list[38]], [ttest_list[10], ttest_list[38]])
        plt.show(block=False)
    return ttest_list, pvalue_list


with open('results t_value.csv', 'w') as f: 
    header = csv.writer(f,['mild', 'yr', 'sept', 'sept+yr', 'mild+yr'])

with open('results p_value.csv', 'w') as f: 
    header = csv.writer(f,['mild', 'yr', 'sept', 'sept+yr', 'mild+yr'])

# def find_average_spectrum(box, cube):
def find_average_spectrum(box):

    im1 = np.mean(box, axis=0)
    im2 = np.mean(im1, axis=0)
    return im2
dic_ROI = {}
dic_ROI['YR'] = []
dic_ROI['Mild'] = []
dic_ROI['YR_Mild'] = []
dic_ROI['Sept_YR'] = []
dic_ROI['Sept'] = []
dic_ROI['*'] = []
area_box_yr, area_box_mild, area_box_yr_mild = [], [], []
list_average_spectrum_mild = []
for k in range(1, 11):

    cube = create_raw_cube('..\Camera_2\Mild\Mild_29_March_Vuka_Rep'+str(k))
    # imshow(cube)
    for box in dic_box_Mild[k]:
       
        spectrum_box = find_average_spectrum(box,cube)
        list_average_spectrum_mild.append(spectrum_box)
        area_box_mild.append((box[0]-box[2])*(box[1]-box[3]))
        
list_ROI = extract_ROI(cube_healthy)
# print(len(list_ROI))
spectrum_healthy = []
for j in range(350):
    # print(int(math.floor(j)))
    subcube_healthy = list_ROI[int(math.floor(j))]
    im1 = np.mean(subcube_healthy, axis = 0)
    im2 = np.mean(im1, axis = 0)
    spectrum_healthy.append(im2)
list_average_spectrum_healthy = np.array(spectrum_healthy)

list_average_spectrum_sept = []
list_average_spectrum_sept_yr = []
area_box_sept_YR, area_box_sept = [], []
for file in os.listdir('../Camera_2/Septoria_YR/label_YR_Sept(field)'):
    mytree = ET.parse('../Camera_2/Septoria_YR/label_YR_Sept(field)/' + file)
    cube = create_raw_cube('../Camera_2/Septoria_YR/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        box = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[box[0]:box[1], box[2]: box[3],:]
        
        if label == 'Sept':
            list_average_spectrum_sept.append(find_average_spectrum(ROI))
            area_box_sept.append((box[1]-box[0])*(box[2]-box[3]))


for file in os.listdir('../Camera_2/Septoria_YR/label_sept_yr'):
    mytree = ET.parse('../Camera_2/Septoria_YR/label_sept_yr/' + file)
    cube = create_raw_cube('../Camera_2/Septoria_YR/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        box = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[box[0]:box[1], box[2]: box[3],:]
        
        if label == 'Sept_YR':
            list_average_spectrum_sept_yr.append(find_average_spectrum(ROI))
            area_box_sept_YR.append((box[1]-box[0])*(box[2]-box[3]))
    
list_average_spectrum_yr_mild = []
area_box_yr_mild = []


for file in os.listdir('../Camera_2/YR_Mild/yr_mild_label'):
    
    mytree = ET.parse('../Camera_2/YR_Mild/yr_mild_label/' + file)
    cube = create_raw_cube('../Camera_2/YR_Mild/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        box = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[box[0]:box[1], box[2]: box[3],:]
        
        if label == 'YR_Mild':
            list_average_spectrum_yr_mild.append(find_average_spectrum(ROI))
            area_box_yr_mild.append((box[1]-box[0])*(box[2]-box[3]))



    list_average_spectrum_yr = []
area_box_yr = []

for file in os.listdir('../Camera_2/YR/yr_label'):
    mytree = ET.parse('../Camera_2/YR/yr_label/' + file)
    cube = create_raw_cube('../Camera_2/YR/' + file[:-4])
    myroot = mytree.getroot()
    # (myroot[1].text[:-4])
    for x in myroot.findall('object'):
        label =x.find('name').text
        box = [int(x.find('bndbox').find('ymin').text), int(x.find('bndbox').find('ymax').text), int(x.find('bndbox').find('xmin').text), int(x.find('bndbox').find('xmax').text)]
        ROI = cube[box[0]:box[1], box[2]: box[3],:]
        
        if label == 'YR':
            list_average_spectrum_yr.append(find_average_spectrum(ROI))
            area_box_yr.append((box[1]-box[0])*(box[2]-box[3]))






tvalue_mild, pvalue_mild = find_band_of_interest(list_average_spectrum_healthy[:len(list_average_spectrum_mild)], list_average_spectrum_mild)
tvalue_yr, pvalue_yr = find_band_of_interest(list_average_spectrum_healthy, list_average_spectrum_yr)
tvalue_sept, pvalue_sept = find_band_of_interest(list_average_spectrum_healthy, list_average_spectrum_sept)
tvalue_sept_yr, pvalue_sept_yr = find_band_of_interest(list_average_spectrum_healthy, list_average_spectrum_sept_yr)
tvalue_yr_mild, pvalue_yr_mild = find_band_of_interest(list_average_spectrum_healthy, list_average_spectrum_yr_mild)

import csv
with open('results t_value.csv', 'a') as f: 
    writer = csv.writer(f)
    for k in range(19):
        writer.writerow([tvalue_mild[k], tvalue_yr[k], tvalue_sept[k], tvalue_sept_yr[k], tvalue_yr_mild[k]])

with open('results p_value.csv', 'a') as f: 
    writer = csv.writer(f)
    for k in range(19):
        writer.writerow([pvalue_mild[k], pvalue_yr[k], pvalue_sept[k], pvalue_sept_yr[k], pvalue_yr_mild[k]])
