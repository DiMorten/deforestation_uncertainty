import numpy as np 
import utils_v1
from icecream import ic
from osgeo import gdal
import pdb
from sklearn.preprocessing._data import _handle_zeros_in_scale

# path_optical_im = 'E:/Jorge/dataset_deforestation/Para_2020/'
# path_label = 'E:/Jorge/dataset_deforestation/Para/'

# dataset = 'Para_2020'
dataset = 'MT_2020'

if dataset == 'Para_2020':
    path_optical_im = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2020/'
    # path_label = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_regeneration/Para/'

    im_filenames = ['S2_PA_2020_07_15_B1_B2_B3.tif',
        'S2_PA_2020_07_15_B4_B5_B6.tif',
        'S2_PA_2020_07_15_B7_B8_B8A.tif',
        'S2_PA_2020_07_15_B9_B10_B11.tif',
        'S2_PA_2020_07_15_B12.tif']
elif dataset == 'MT_2020':
    path_optical_im = 'D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/MG_10m/S2/2020/'
    # path_label = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_regeneration/Para/'

    im_filenames = ['S2_R1_MT_2020_08_03_2020_08_15_B1_B2.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B3_B4.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B5_B6.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B7_B8.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B8A_B9.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B10_B11.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B12.tif']    

        
def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    im = gdal_header.ReadAsArray()
    return im

def loadOpticalIm(im_filenames):
    band_count = 0

    for i, im_filename in enumerate(im_filenames):
        ic(path_optical_im + im_filename)        
        im = load_tiff_image(path_optical_im + im_filename).astype('float32')
        ic(im.shape)
        if len(im.shape) == 2: im = im[np.newaxis, ...]
        if i:
            ic(im.shape, optical_im.shape)
            optical_im = np.concatenate((optical_im, im), axis=0)
        else:
            optical_im = im
    del im 
    return optical_im    
def exclude60mBands(optical_im):
    sentinel2_band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
            'B8A', 'B9', 'B10', 'B11', 'B12']
    no60m_ids = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    sentinel_2band_names_no60m = [sentinel2_band_names[x] for x in no60m_ids]
    ic(sentinel_2band_names_no60m)
    return optical_im[..., no60m_ids]
def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)] = np.mean(img) # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    min_value, max_value = [], []
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(), bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        min_value.append(hist[1][len(cum_hist[cum_hist<bth])])
        max_value.append(hist[1][len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]



createTif = True

if createTif == True:
    optical_im = loadOpticalIm(im_filenames)
    optical_im = np.transpose(optical_im, (1, 2, 0))
    optical_im = exclude60mBands(optical_im)
    ic(optical_im.shape)
    np.save('optical_im_unnormalized.npy', optical_im)
else:
    optical_im = np.load('optical_im_unnormalized.npy') 
 
class NormalizationManager():
    def __init__(self, img, feature_range=[0, 1]):

        self.feature_range = feature_range
        self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.02, uth=0.98)
        self.min_val = np.nanmin(img, axis=(0,1))
        self.max_val = np.nanmax(img, axis=(0,1))

        self.min_val = np.clip(self.min_val, self.clips[0], None)
        self.max_val = np.clip(self.max_val, None, self.clips[1])
    def clip_image(self, img):
        return np.clip(img.copy(), self.clips[0], self.clips[1])

    def normalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale
        
        img = self.clip_image(img.copy())
        img *= scale
        img += min_
        return img

normalizationManager = NormalizationManager(optical_im)
optical_im = normalizationManager.normalize(optical_im)
ic(np.min(optical_im), np.average(optical_im), np.max(optical_im))

# pdb.set_trace()
np.save(path_optical_im + 'optical_im.npy', optical_im) 
# plt.imshow(optical_im[...,[2,1,0]])
# plt.show()
