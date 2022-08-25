import numpy as np 
import utils_v1
from icecream import ic
from osgeo import gdal
import pdb
from sklearn.preprocessing._data import _handle_zeros_in_scale
from src.dataset import (
    Para, MT,
    ParaMultipleDates, MTMultipleDates
)
# path_image_unnormalized = 'E:/Jorge/dataset_deforestation/Para_2020/'
# path_label = 'E:/Jorge/dataset_deforestation/Para/'

# dataset = 'Para_2020'
# dataset = 'Para_2020'
# dataset = 'MT_2019_2020'
# dataset = 'MT_2020'
dataset = 'Para_2015'
# dataset = 'Para_2016'
# dataset = 'Para_2017'
# dataset = 'Para_2018'
# dataset = 'Para_2019'

# dataset = 'Para_2018_2019'
# dataset = 'MT_2019_2020'

# dataset = 'MT_2020'
# dataset = 'MT_2016'
ic(dataset)
mask_input = 'deforestation_time'

if dataset == 'Para_2020':
    if mask_input == 'deforestation_time':
        dataset = ParaMultipleDates()
        path_image_unnormalized = dataset.paths.deforestation_past_years # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2020.npy'
        im_filenames = ['deforestation_past_years.tif']
        latest_year = 2020

elif dataset == 'Para_2019':
    if mask_input == 'deforestation_time':
        dataset = ParaMultipleDates()
        path_image_unnormalized = dataset.paths.deforestation_past_years # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2019.npy'
        # im_filenames = ['deforestation_past_years.tif']
        im_filenames = ['']
        latest_year = 2019


elif dataset == 'Para_2018':
    if mask_input == 'deforestation_time':
        dataset = ParaMultipleDates()
        path_image_unnormalized = dataset.paths.deforestation_past_years # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2018.npy'
        im_filenames = ['']
        latest_year = 2018

elif dataset == 'Para_2017':
    if mask_input == 'deforestation_time':
        dataset = ParaMultipleDates()
        path_image_unnormalized = dataset.paths.deforestation_past_years # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2017.npy'
        im_filenames = ['deforestation_past_years.tif']
        latest_year = 2017

elif dataset == 'Para_2016':
    if mask_input == 'deforestation_time':
        dataset = ParaMultipleDates()
        path_image_unnormalized = dataset.paths.deforestation_past_years # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2016.npy'
        im_filenames = ['deforestation_past_years.tif']
        latest_year = 2016

elif dataset == 'Para_2015':
    if mask_input == 'deforestation_time':
        dataset = ParaMultipleDates()
        path_image_unnormalized = dataset.paths.deforestation_past_years # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2015.npy'
        im_filenames = ['deforestation_past_years.tif']
        latest_year = 2015

elif dataset == 'Para_2018_2019':
    if mask_input == 'deforestation_time':
        dataset = Para()
        path_image_unnormalized = dataset.paths.label # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2018_2019.npy'
        im_filenames = ['deforestation_past_years.tif']  
        latest_year = 2018
elif dataset == 'MT_2019_2020':
    if mask_input == 'deforestation_time':
        dataset = MT()
        path_image_unnormalized = dataset.paths.label # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2019_2020.npy'
        im_filenames = ['deforestation_past_years.tif']  
        latest_year = 2019
elif dataset == 'MT_2020':
    if mask_input == 'deforestation_time':
        dataset = MT()
        path_image_unnormalized = dataset.paths.label # 'D:/Jorge/datasets/deforestation/Para/'
        im_filename_normalized = 'deforestation_time_normalized_2020.npy'
        im_filenames = ['deforestation_past_years.tif']  
        latest_year = 2020
elif dataset == 'MT_2018':
    if mask_input == 'deforestation_time':
        dataset = MT()
        path_image_unnormalized = dataset.paths.label 
        im_filename_normalized = 'deforestation_time_normalized_2018.npy'
        im_filenames = ['deforestation_past_years.tif']  
        latest_year = 2018
elif dataset == 'MT_2017':
    if mask_input == 'deforestation_time':
        dataset = MT()
        path_image_unnormalized = dataset.paths.label 
        im_filename_normalized = 'deforestation_time_normalized_2017.npy'
        im_filenames = ['deforestation_past_years.tif']  
        latest_year = 2017
elif dataset == 'MT_2016':
    if mask_input == 'deforestation_time':
        dataset = MT()
        path_image_unnormalized = dataset.paths.label 
        im_filename_normalized = 'deforestation_time_normalized_2016.npy'
        im_filenames = ['deforestation_past_years.tif']  
        latest_year = 2016

def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    im = gdal_header.ReadAsArray()
    return im

def loadOpticalIm(im_filenames):
    band_count = 0

    for i, im_filename in enumerate(im_filenames):
        ic(path_image_unnormalized)        
        im = load_tiff_image(path_image_unnormalized).astype('float32')
        ic(im.shape)
        if len(im.shape) == 2: im = im[np.newaxis, ...]
        if i:
            ic(im.shape, image_unnormalized.shape)
            image_unnormalized = np.concatenate((image_unnormalized, im), axis=0)
        else:
            image_unnormalized = im
    del im 
    return image_unnormalized    
def exclude60mBands(image_unnormalized):
    sentinel2_band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
            'B8A', 'B9', 'B10', 'B11', 'B12']
    no60m_ids = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    sentinel_2band_names_no60m = [sentinel2_band_names[x] for x in no60m_ids]
    ic(sentinel_2band_names_no60m)
    return image_unnormalized[..., no60m_ids]
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
    image_unnormalized = loadOpticalIm(im_filenames)
    ic(image_unnormalized.shape)
    # pdb.set_trace()
    image_unnormalized = np.transpose(image_unnormalized, (1, 2, 0))
    ic(image_unnormalized.shape)
    if mask_input == 'deforestation_time':
        deforestation_areas = image_unnormalized.copy()
        deforestation_areas[image_unnormalized != 0] = 1
        ic(np.unique(image_unnormalized, return_counts=True))

        print("calculate time distance to latest year " + str(latest_year))
        image_unnormalized = latest_year - image_unnormalized
        ic(np.unique(image_unnormalized, return_counts=True))

        print("negative values are after the latest year. Set to 0")
        image_unnormalized[image_unnormalized<0] = -1
        ic(np.unique(image_unnormalized, return_counts=True))


        print("add 1 for latest years areas to be equal to 1")
        image_unnormalized = image_unnormalized + 1
        ic(np.unique(image_unnormalized, return_counts=True))


        print("set non deforestated areas to 0")
        image_unnormalized[deforestation_areas == 0] = 0

        ic(np.unique(image_unnormalized, return_counts=True))
        print("adding deforestation before 2008")
        label_past_deforestation_before_2008 = dataset.loadPastDeforestationBefore2008()
        deforestation_time_2008 = np.max(image_unnormalized)
        ic(deforestation_time_2008)
        image_unnormalized[label_past_deforestation_before_2008 == 2007] = deforestation_time_2008 + 1
        # image_unnormalized[label_past_deforestation_before_2008 == 1] = 2007

        ic(np.unique(image_unnormalized, return_counts=True))

        '''
        label_past_deforestation = dataset.loadPastDeforestationLabel()
        deforestation_before_2008 = label_past_deforestation.copy()
        deforestation_before_2008[np.squeeze(image_unnormalized)>deforestation_time_2008] = 0
        ic(np.unique(deforestation_before_2008, return_counts=True))
        image_unnormalized[deforestation_before_2008 == 1] = deforestation_time_2008 + 1
        '''
        # ic(np.unique(image_unnormalized, return_counts=True))
    '''
    if subtractYears == True:
        ic(np.unique(image_unnormalized, return_counts=True))
        image_unnormalized = image_unnormalized - 2
        image_unnormalized[image_unnormalized<0] = 0
        ic(np.unique(image_unnormalized, return_counts=True))
    '''
    # ic(image_unnormalized.shape)
    # np.save('optical_im_unnormalized.npy', image_unnormalized)
else:
    pass
    # image_unnormalized = np.load('optical_im_unnormalized.npy') 
 
class NormalizationManager():
    def __init__(self, img, feature_range=[0, 1]):

        self.feature_range = feature_range
        # self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.02, uth=0.98)
        self.min_val = np.nanmin(img, axis=(0,1))
        self.max_val = np.nanmax(img, axis=(0,1))

        # self.min_val = np.clip(self.min_val, self.clips[0], None)
        # self.max_val = np.clip(self.max_val, None, self.clips[1])
    def clip_image(self, img):
        return np.clip(img.copy(), self.clips[0], self.clips[1])

    def normalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale
        
        # img = self.clip_image(img.copy())
        img *= scale
        img += min_
        return img

normalizationManager = NormalizationManager(image_unnormalized)
image_unnormalized = normalizationManager.normalize(image_unnormalized)
ic(np.min(image_unnormalized), np.average(image_unnormalized), np.max(image_unnormalized))
ic(np.unique(image_unnormalized, return_counts=True))
# pdb.set_trace()
print("Saving to... ", path_image_unnormalized + im_filename_normalized)
np.save(path_image_unnormalized + im_filename_normalized, image_unnormalized) 
# plt.imshow(image_unnormalized[...,[2,1,0]])
# plt.show()
