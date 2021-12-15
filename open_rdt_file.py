import hsd as hs
from statsmodels.graphics.tsaplots import plot_acf, acf, pacf
# inpdict = hs.load("test3_green_leaves.hsd")
import rawpy
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

h, w = 300, 640
data = np.zeros((h, w, 3), dtype=np.uint8)




path = "test3_green_leaves camera2.rdt"

# data_before_process = np.fromfile(path, dtype=np.uint16)
# data_without_0 = list(filter(lambda a: a != 0, data_before_process))
# print(data_before_process.shape)
# print(h*w*456)

f = open(path, 'r')
k= 0


for line in f.readlines(): 
    if k == 2: 
        data_before_process_str = line.split()
    k = k+1
    
# print(data_before_process[:100])
# print(len(data_before_process))
data_before_process = []
for j in range(len(data_before_process_str)):
    data_before_process.append(float(data_before_process_str[j]))

max_intensity = np.max(data_before_process)
print(max_intensity)

def RGB_treatment():
    data_2 = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):

            data_2[i][j] = [data_before_process[j + w*456*i+159*w], data_before_process[j + w*456*i+146*w], data_before_process[j + w*456*i+126*w]]
            # data_2[i][j] = [data_before_process[j + w*456*i+97*w], data_before_process[j + w*456*i+85*w], data_before_process[j + w*456*i+75*w]]
            # data_2[i][j] = [data_before_process[j + w*456*i], 0, 0]

    return data_2

def display_image_for_one_wave_length(lambda_var):

    data_2 = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # data_2[i][j] = [data_before_process[j + w*98*i+72*w], data_before_process[j + w*98*i+49*w], data_before_process[j + w*98*i+24*w]]
            data_2[i][j] = data_before_process[j + w*456*i+lambda_var*w]
    # fig = plt.figure(figsize = (w, h))
    fig = plt.figure()
    X, Y = np.meshgrid(np.arange(h), np.arange(w))
    levels = MaxNLocator(nbins=15).tick_values(0, max_intensity)
    cmap = plt.colormaps['PiYG']
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    pc = plt.pcolormesh(np.transpose(X), np.transpose(Y), data_2, vmin = 0, vmax = max_intensity)
    fig.colorbar(pc)
    plt.title('image lambda = ' + 'nm' )
    plt.show()
    plt.savefig('image_one_lambda')


img = Image.fromarray(RGB_treatment(), 'RGB')
img.save('my.png')
img.show()
# print(data_before_process[len(data_before_process)-250:])

def plot_autocorrelation():
    # autocorrelation_image = acf(data_before_process[100000:200000], nlags = 5000)
    autocorrelation_image = acf(data_before_process[100000:200000], nlags = 500 + 702*20)

    plt.plot(np.arange(501 + 702*20), autocorrelation_image)
    plt.show()
    plt.savefig("autocorrelation image")
    
display_image_for_one_wave_length(146)

# plot_autocorrelation()


#R : 159 G: 146 B: 126
