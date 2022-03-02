import rawpy
import imageio
import numpy as np
from PIL import Image
from spectral import *
import spectral.io.aviris as aviris
import wx
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from scipy.signal import argrelextrema


spectral.settings.WX_GL_DEPTH_SIZE = 16
path = 'image_1.raw'

h, w = 952, 2048
band = 98
data_before_process = np.fromfile(path, dtype=np.uint16)

data_1 = np.reshape(data_before_process, (h, w*band))
# data_2 = np.reshape(data_1[:,], )


def create_cube():
    cube = np.zeros((h, w, band), dtype=np.uint8)
    for k in range(h):
        cube[k,:,:] = np.transpose(np.reshape(data_1[k,:], (band, w)))
    return cube

cube = create_cube()
# view_cube(cube, bands = [29,19,9])

view = imshow(cube, (49, 29, 9))

def create_cropped_cube(cube_of_interest):
    cropped_cube = np.zeros((h,w,band))
    vi = ndvi(cube_of_interest, 25, 49)
    vi_mask = np.where(vi < 0.3, 0, 1)
    for k in range(band):
        cropped_cube[:,:,k] = np.multiply(cube_of_interest[:,:,k],vi_mask)
    return cropped_cube

def find_interesting_band(cropped_cube_of_interest): 
    var_list = []
    for k in range(band):
        var_list.append(np.var(cropped_cube_of_interest[:,:,k]))
    dvar = np.gradient(var_list, 1)
    print(argrelextrema(dvar, np.greater))
    return dvar

cropped_cube = create_cropped_cube()
dim_reduc_cub = cropped_cube[:,:,[43, 51, 61 ]]

def average_spectra_region(region, cropped_cube_of_interest): 
    spectra = []
    for k in range(band) :
        region_of_interest = cropped_cube_of_interest[region[0]:region[1], region[2]:region[3],k]
        region_of_interest_non_zero = region_of_interest[np.nonzero(region_of_interest)]
        spectra.append(np.mean(region_of_interest_non_zero))
    return spectra

# print(np.shape(cube))
# cube_RGB = cube[:,:,(60,70,80)]
# img = Image.fromarray(cube_RGB, "RGB")
# img.save('my.png')
# img.show()

#interesting bands: 44, 53, 65