import rawpy
import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


h, w = 696, 2048
data = np.zeros((h, w, 3), dtype=np.uint8)


path = 'test3_green_leaves.raw'


data_before_process = np.fromfile(path, dtype=np.uint16)
# data = np.reshape(data, (height, width))  

print(len(data_before_process)==h*w*98)
max_intensity = (np.max(data_before_process))

lambda_list = [
 601.00,
 602.00,
 606.00,
 610.00,
 612.36,
 616.73,
 621.20,
 624.07,
 626.71,
 628.05,
 631.63,
 636.59,
 641.24,
 646.58,
 651.24,
 654.85,
 658.63,
 660.78,
 665.41,
 669.57,
 673.58,
 683.94,
 688.45,
 692.56,
 696.90,
 699.05,
 703.90,
 708.03,
 712.41,
 719.38,
 724.03,
 727.76,
 731.65,
 734.10,
 737.94,
 741.27,
 744.62,
 752.29,
 758.12,
 762.98,
 767.84,
 770.30,
 775.42,
 780.03,
 784.55,
 790.34,
 795.45,
 799.87,
 804.17,
 806.41,
 810.84,
 814.31,
 817.33,
 827.88,
 832.47,
 836.32,
 840.31,
 842.50,
 846.97,
 850.85,
 854.61,
 859.38,
 863.32,
 866.84,
 870.24,
 872.16,
 875.86,
 879.01,
 881.74,
 886.01,
 889.95,
 893.54,
 897.35,
 899.58,
 903.55,
 906.80,
 909.86,
 914.24,
 917.38,
 920.24,
 923.17,
 924.90,
 928.03,
 930.66,
 933.14,
 942.38,
 945.32,
 947.88,
 950.70,
 952.39,
 955.99,
 959.30,
 962.35,
 966.54,
 969.13,
 971.34,
 973.58,
 975.08,
]



def RGB_treatment():
    # RGB index = [25, 10, 0] for the wavelength [699.05, 631.63, 601.0]
    data_2 = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            data_2[i][j] = [data_before_process[j + w*98*i+25*w], data_before_process[j + w*98*i+10*w], data_before_process[j + w*98*i+0*w]]
            # data_2[i][j] = [data_before_process[j + w*98*i+97*w], data_before_process[j + w*98*i+85*w], data_before_process[j + w*98*i+75*w]]

            # data_2[i][j] = [0,0, data_before_process[j + w*98*i]]
           
    return data_2

def display_image_for_one_wave_length(lambda_var):

    data_2 = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # data_2[i][j] = [data_before_process[j + w*98*i+72*w], data_before_process[j + w*98*i+49*w], data_before_process[j + w*98*i+24*w]]
            data_2[i][j] = data_before_process[j + w*98*i+lambda_var*w]
    # fig = plt.figure(figsize = (w, h))
    fig = plt.figure()
    X, Y = np.meshgrid(np.arange(h), np.arange(w))
    levels = MaxNLocator(nbins=15).tick_values(0, max_intensity)
    cmap = plt.colormaps['PiYG']
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    pc = plt.pcolormesh(np.transpose(X), np.transpose(Y), data_2, vmin = 0, vmax = 606)
    fig.colorbar(pc)
    plt.title('image lambda = ' + str(np.round(lambda_list[lambda_var],1)) + 'nm' )
    plt.show()
    plt.savefig('image_one_lambda')


def display_RGB_image():
    img = Image.fromarray(RGB_treatment(), 'RGB')
    img.save('my.png')
    img.show()


# plt.plot(np.arange(2048*4), data_before_process[: 2048*4])
# plt.title("Spectra of the first frame, 4 lambda")
# plt.savefig("spectra first frame")
# for i in [25, 40, 70, 96]:

print(lambda_list[97], lambda_list[80], lambda_list[75])
# display_image_for_one_wave_length(97)
