import numpy as np 
import utils_v1
from icecream import ic
from osgeo import gdal
import pdb
from sklearn.preprocessing._data import _handle_zeros_in_scale
import cv2
import src.rasterTools as rasterTools

from src.dataset import PA, PADeforestationTime, PADistanceMap, PAMultipleDates, MTMultipleDates, MT, MA

# ======= INPUT PARAMETERS ============ # 
dataset = MA()
year = 2020
maskOutClouds = True
# ======= END INPUT PARAMETERS ============ # 

scale_list = None
exclude60mBandsFlag = True

path_optical_im = dataset.paths.optical_im_past_dates[year]

if type(dataset) == MA:
    resolution_list = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]
    scale_list = [x/10 for x in resolution_list]
    ic(scale_list)

 
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

    optical_im = rasterTools.loadOpticalIm(path_optical_im, dataset.paths.im_filenames[year], scale_list)
    ic(optical_im.shape)
    # pdb.set_trace()
    optical_im = np.transpose(optical_im, (1, 2, 0))
    ic(optical_im.shape)
    if exclude60mBandsFlag == True:
        optical_im = rasterTools.exclude60mBands(optical_im)
    ic(optical_im.shape)
    # ic(np.min(optical_im), np.mean(optical_im), np.max(optical_im))
    # ic(np.count_nonzero(~np.isnan(optical_im)), np.count_nonzero(np.isnan(optical_im)))
    if maskOutClouds == True:
        cloud_cloudshadow_mask = np.load(dataset.paths.cloud_mask[year]).astype(np.uint8)
        cloud_mask = np.zeros_like(cloud_cloudshadow_mask)
        cloud_mask[cloud_cloudshadow_mask == 1] = 2
        del cloud_cloudshadow_mask
    else:
        cloud_mask = np.zeros(optical_im.shape[:-1], dtype=np.uint8)
    cloud_mask[np.isnan(optical_im[...,0])] = 2
    cloud_mask[optical_im[...,0] == 0] = 2

    '''
    plt.figure(figsize = (12,12))
    plt.axis('off')
    plt.imshow(cloud_mask)
    plt.savefig('normalization_mask.png', dpi=200, bbox_inches='tight')
    pdb.set_trace()
    '''

    optical_im = np.nan_to_num(optical_im)

    # ic(np.min(optical_im), np.mean(optical_im), np.max(optical_im))
    # ic(np.count_nonzero(~np.isnan(optical_im)), np.count_nonzero(np.isnan(optical_im)))
    # np.save('optical_im_unnormalized.npy', optical_im)
else:
    pass
    # optical_im = np.load('optical_im_unnormalized.npy') 
 
class NormalizationManager():
    def __init__(self, img, mask=[0], feature_range=[0, 1]):

        self.feature_range = feature_range
        self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.02, uth=0.98, mask=mask)
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

# normalizationManager = NormalizationManager(optical_im, cloud_mask)
normalizationManager = NormalizationManager(optical_im, cloud_mask)
optical_im = normalizationManager.normalize(optical_im)
ic(np.min(optical_im), np.average(optical_im), np.max(optical_im))

# pdb.set_trace()
np.save(path_optical_im + 'optical_im.npy', optical_im) 
# plt.imshow(optical_im[...,[2,1,0]])
# plt.show()
