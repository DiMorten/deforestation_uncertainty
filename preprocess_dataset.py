import numpy as np 
import utils_v1


path_optical_im = 'E:/Jorge/dataset_deforestation/Para_2020/'
path_label = 'E:/Jorge/dataset_deforestation/Para/'



im = utils_v1.load_optical_image(path_optical_im + 'S2_PA_2020_07_15_B1_B2_B3.tif')
print(im.shape)
