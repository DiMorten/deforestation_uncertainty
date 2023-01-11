import numpy as np 
import scipy 
import scipy.signal as scisig 
from icecream import ic 
import pdb 
import matplotlib.pyplot as plt 
import rasterio 
from osgeo import gdal 
import cv2 
import src.rasterTools as rasterTools

from src.dataset import (
    PA, PADeforestationTime, PADistanceMap, PAMultipleDates, MTMultipleDates,
    MT, 
    MA
)
# naming conventions: 
# ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12'] 
# ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2']) 
# [        1,    2,      3,       4,     5,    6,    7,    8,     9,      10,            11,      12,     13]) #gdal 
# [        0,    1,      2,       3,     4,    5,    6,    7,     8,      9,            10,      11,     12]) #numpy 
# [              BB      BG       BR                       BNIR                                  BSWIR1    BSWIR2 
 
# ge. Bands 1, 2, 3, 8, 11, and 12 were utilized as BB , BG , BR , BNIR , BSWIR1 , and BSWIR2, respectively. 
 
def get_rescaled_data(data, limits): 
    return (data - limits[0]) / (limits[1] - limits[0]) 
 
 
def get_normalized_difference(channel1, channel2): 
    subchan = channel1 - channel2 
    sumchan = channel1 + channel2 
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions 
    return subchan / sumchan 
 
 
def get_shadow_mask(data_image): 
    # get data between 0 and 1 
    data_image = data_image / 10000. 
 
    (ch, r, c) = data_image.shape 
    shadow_mask = np.zeros((r, c)).astype('float32') 
 
    BB = data_image[1] 
    BNIR = data_image[7] 
    BSWIR1 = data_image[11] 
 
    CSI = (BNIR + BSWIR1) / 2. 
 
    t3 = 3 / 4 
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI)) 
 
    t4 = 5 / 6 
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB)) 
 
    shadow_tf = np.logical_and(CSI < T3, BB < T4) 
 
    shadow_mask[shadow_tf] = -1 
    shadow_mask = scisig.medfilt2d(shadow_mask, 5) 
 
    return shadow_mask 
 
 
def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False): 
    data_image = data_image / 10000. 
    (ch, r, c) = data_image.shape 
 
    # Cloud until proven otherwise 
    score = np.ones((r, c)).astype('float32') 
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands. 
    score = np.minimum(score, get_rescaled_data(data_image[1], [0.1, 0.5])) 
    score = np.minimum(score, get_rescaled_data(data_image[0], [0.1, 0.3])) 
    score = np.minimum(score, get_rescaled_data((data_image[0] + data_image[10]), [0.15, 0.2])) 
    # Clouds are reasonably bright in all visible bands. 
    score = np.minimum(score, get_rescaled_data((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8])) 
 
    if use_moist_check: 
        # Clouds are moist 
        ndmi = get_normalized_difference(data_image[7], data_image[11]) 
        score = np.minimum(score, get_rescaled_data(ndmi, [-0.1, 0.1])) 
 
    # However, clouds are not snow. 
    ndsi = get_normalized_difference(data_image[2], data_image[11]) 
    score = np.minimum(score, get_rescaled_data(ndsi, [0.8, 0.6])) 
 
    box_size = 7 
    box = np.ones((box_size, box_size)) / (box_size ** 2) 
    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5)) 
    score = scisig.convolve2d(score, box, mode='same') 
 
    score = np.clip(score, 0.00001, 1.0) 
 
    if binarize: 
        score[score >= cloud_threshold] = 1 
        score[score < cloud_threshold] = 0 
 
    return score 
 
 
def get_cloud_cloudshadow_mask(data_image, cloud_threshold = 0.2): 
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True) 
    shadow_mask = get_shadow_mask(data_image) 
 
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask) 
    cloud_cloudshadow_mask[shadow_mask < 0] = -1 
    cloud_cloudshadow_mask[cloud_mask > 0] = 1 
     
    #pdb.set_trace() 
    return cloud_cloudshadow_mask 
 
 
if __name__ == '__main__': 
    # ======= INPUT PARAMETERS ============ # 
    dataset = MA()
    year = 2021

    addThinCloudMask = False
    apply_shadow_mask = False
    # ======= END INPUT PARAMETERS ============ # 

    path_optical_im = dataset.paths.optical_im_past_dates[year]

    scale_list = None

    if type(dataset) == PA:
        
        path_cirrus_band = dataset.paths.im_filenames[year][3]
        cirrus_band_id = 1

    elif type(dataset) == MT: 

        path_cirrus_band = dataset.paths.im_filenames[year][5]
        cirrus_band_id = 0

    elif type(dataset) == MA:

        path_cirrus_band = dataset.paths.im_filenames[year][10]
        cirrus_band_id = None

        resolution_list = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 60, 20, 20]
        scale_list = [x/10 for x in resolution_list]
        ic(scale_list)

    dataset_id = dataset.__class__.__name__ + '_' + str(year)
    filename = dataset_id + '.npy' 
 
    optical_im = rasterTools.loadOpticalIm(path_optical_im, dataset.paths.im_filenames[year], scale_list) 
    # optical_im = np.transpose(optical_im, (1, 2, 0)) 
    ic(optical_im.shape) 
 
    # === GET CLOUD CLOUD-SHADOW MASK === #
 
    cloud_cloudshadow_mask = get_cloud_cloudshadow_mask(optical_im, cloud_threshold = 0.2).astype(np.int8) 

    ic(cloud_cloudshadow_mask.shape) 
    ic(np.unique(cloud_cloudshadow_mask, return_counts = True)) 

 
    plt.figure() 
    plt.imshow(cloud_cloudshadow_mask) 
    plt.axis('off') 
    plt.savefig(path_optical_im + 'cloud_cloudshadow_mask_' + filename + '.png', dpi = 500)

    # === GET CIRRUS THIN CLOUD MASK === #
    ic(path_optical_im + path_cirrus_band)
    cirrus = rasterTools.load_tiff_image(path_optical_im + path_cirrus_band)[cirrus_band_id]
    if scale_list != None:
        cirrus = rasterTools.scaleIm(cirrus, 6.0)
    ic(np.min(cirrus), np.average(cirrus), np.max(cirrus))
    threshold = 19

    thin_cloud_mask = np.zeros_like(cirrus).astype(np.uint8)
    thin_cloud_mask[cirrus > threshold] = 1

    # === SAVE CLOUD MASK === #

    cloud_mask = np.zeros_like(cirrus).astype(np.uint8)
    
    if addThinCloudMask == True:
        cloud_mask[thin_cloud_mask == 1] = 1
    ic(cloud_mask.shape, cloud_cloudshadow_mask.shape, thin_cloud_mask.shape, cirrus.shape)
    # pdb.set_trace()
    cloud_mask[cloud_cloudshadow_mask == 1] = 1
    
    if apply_shadow_mask == True:
        cloud_mask[cloud_cloudshadow_mask == -1] = 1

    plt.figure() 
    plt.imshow(cloud_mask) 
    plt.axis('off') 
    plt.savefig(path_optical_im + 'cloudmask_' + filename + '.png', dpi = 500)
    

    print("saving in... " + path_optical_im + "cloudmask_" + filename)
    np.save(path_optical_im + "cloudmask_" + filename, cloud_mask) 

