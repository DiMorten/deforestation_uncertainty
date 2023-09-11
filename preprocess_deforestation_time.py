import os
import numpy as np 
import utils_v1
from icecream import ic
from osgeo import gdal
import pdb
import src.rasterTools as rasterTools

from sklearn.preprocessing._data import _handle_zeros_in_scale
from src.dataset import (
    PA, MT, MA,
    PAMultipleDates, MTMultipleDates, MAMultipleDates, MSMultipleDates, PIMultipleDates, MOMultipleDates,
    L8MTMultipleDates, L8AMMultipleDates
)

ic.configureOutput(includeContext=True)
# ======= INPUT PARAMETERS ============ #
config = {
    'dataset': 'L8AM',
    'year': 2022, # latest year
}
mask_input = 'deforestation_time'
loadDeforestationBefore2008Flag = True
# ======= END INPUT PARAMETERS ============ #

latest_year = config['year']

if config['dataset'] == 'PA':
    dataset = PAMultipleDates()
elif config['dataset'] == 'MT':
    dataset = MTMultipleDates()
elif config['dataset'] == 'MA':
    dataset = MAMultipleDates()
elif config['dataset'] == 'MS':
    dataset = MSMultipleDates()
elif config['dataset'] == 'PI':
    dataset = PIMultipleDates()
elif config['dataset'] == 'MO':
    dataset = MOMultipleDates()
elif config['dataset'] == 'L8MT':
    dataset = L8MTMultipleDates()
elif config['dataset'] == 'L8AM':
    dataset = L8AMMultipleDates()

ic(dataset)


if mask_input == 'deforestation_time':
    out_filename = 'deforestation_time_normalized_{}.npy'.format(latest_year)
    
'''
if dataset == 'MA' and latest_year == 2020:
    if mask_input == 'deforestation_time':
        dataset = MA()
        path_image_unnormalized = dataset.paths.deforestation_past_years
        im_filename_normalized = 'deforestation_time_normalized_{}.npy'.format(latest_year)
        im_filenames = ['']  
        latest_year = 2020
        loadDeforestationBefore2008Flag = False
'''
ic(dataset)
 
    
def exclude60mBands(image_unnormalized):
    sentinel2_band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
            'B8A', 'B9', 'B10', 'B11', 'B12']
    no60m_ids = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    sentinel_2band_names_no60m = [sentinel2_band_names[x] for x in no60m_ids]
    ic(sentinel_2band_names_no60m)
    return image_unnormalized[..., no60m_ids]


createTif = True
if createTif == True:
    deforestation_time = rasterTools.load_tiff_image(dataset.paths.deforestation_past_years)
    ic(deforestation_time.shape)
    # pdb.set_trace()
    deforestation_time = np.expand_dims(deforestation_time, axis=-1)
    ic(deforestation_time.shape)
    if mask_input == 'deforestation_time':
        deforestation_areas = deforestation_time.copy()
        deforestation_areas[deforestation_time != 0] = 1
        ic(np.unique(deforestation_time, return_counts=True))

        print("calculate time distance to latest year " + str(latest_year))
        deforestation_time = latest_year - deforestation_time
        ic(np.unique(deforestation_time, return_counts=True))

        print("negative values are after the latest year. Set to 0")
        deforestation_time[deforestation_time<0] = -1
        ic(np.unique(deforestation_time, return_counts=True))


        print("add 1 for latest years areas to be equal to 1")
        deforestation_time = deforestation_time + 1
        ic(np.unique(deforestation_time, return_counts=True))


        print("set non deforestated areas to 0")
        deforestation_time[deforestation_areas == 0] = 0

        ic(np.unique(deforestation_time, return_counts=True))
        if loadDeforestationBefore2008Flag == True:
            print("adding deforestation before 2008")
            label_past_deforestation_before_2008 = dataset.loadPastDeforestationBefore2008()
            deforestation_time_2008 = np.max(deforestation_time)
            ic(deforestation_time_2008)
            deforestation_time[label_past_deforestation_before_2008 != 0] = deforestation_time_2008 + 1
            # deforestation_time[label_past_deforestation_before_2008 == 1] = 2007

        ic(np.unique(deforestation_time, return_counts=True))

        '''
        label_past_deforestation = dataset.loadPastDeforestationLabel()
        deforestation_before_2008 = label_past_deforestation.copy()
        deforestation_before_2008[np.squeeze(deforestation_time)>deforestation_time_2008] = 0
        ic(np.unique(deforestation_before_2008, return_counts=True))
        deforestation_time[deforestation_before_2008 == 1] = deforestation_time_2008 + 1
        '''
        # ic(np.unique(deforestation_time, return_counts=True))
    '''
    if subtractYears == True:
        ic(np.unique(deforestation_time, return_counts=True))
        deforestation_time = deforestation_time - 2
        deforestation_time[deforestation_time<0] = 0
        ic(np.unique(deforestation_time, return_counts=True))
    '''
    # ic(deforestation_time.shape)
    # np.save('optical_im_unnormalized.npy', deforestation_time)
else:
    pass
 
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

normalizationManager = NormalizationManager(deforestation_time)
deforestation_time = normalizationManager.normalize(deforestation_time)
ic(np.min(deforestation_time), np.average(deforestation_time), np.max(deforestation_time))
ic(np.unique(deforestation_time, return_counts=True))
# pdb.set_trace()

folder = '/'.join((dataset.paths.deforestation_past_years).split('/')[:-1]) + '/'
print("Saving to... ", os.path.join(folder, out_filename))

np.save(os.path.join(folder, out_filename), deforestation_time) 
# plt.imshow(image_unnormalized[...,[2,1,0]])
# plt.show()
