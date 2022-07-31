import numpy as np 
import scipy 
import scipy.signal as scisig 
from icecream import ic 
import pdb 
import matplotlib.pyplot as plt 
import rasterio 
from osgeo import gdal 
import cv2 
from src.dataset import (
    Para, ParaDeforestationTime, ParaDistanceMap, ParaMultipleDates, MTMultipleDates,
    MT, 
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
 
def load_tiff_image(path): 
    # Read tiff Image 
    print (path)  
    gdal_header = gdal.Open(path) 
    im = gdal_header.ReadAsArray() 
    return im 
def loadOpticalIm(path_optical_im, im_filenames): 
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
 
 
if __name__ == '__main__': 
 
    dataset_id = 'MT_2020' 

    if dataset_id == 'Para_2015': 
        dataset = Para() 
        path_optical_im = dataset.paths.optical_im_folder + 'Para_2015/' 
        im_filenames = ['PA_S2_2015_B1_B2_B3_crop.tif', 
            'PA_S2_2015_B4_B5_B6_crop.tif', 
            'PA_S2_2015_B7_B8_B8A_crop.tif', 
            'PA_S2_2015_B9_B10_B11_crop.tif', 
            'PA_S2_2015_B12_crop.tif'] 
        path_cirrus_band = im_filenames[3]
        cirrus_band_id = 1

    if dataset_id == 'Para_2016': 
        dataset = Para() 
        path_optical_im = dataset.paths.optical_im_folder + 'Para_2016/' 
        im_filenames = ['PA_S2_2016_B1_B2_B3_crop.tif', 
            'PA_S2_2016_B4_B5_B6_crop.tif', 
            'PA_S2_2016_B7_B8_B8A_crop.tif', 
            'PA_S2_2016_B9_B10_B11_crop.tif', 
            'PA_S2_2016_B12_crop.tif'] 
        path_cirrus_band = im_filenames[3]
        cirrus_band_id = 1
   
    if dataset_id == 'Para_2017': 
        dataset = Para() 
        path_optical_im = dataset.paths.optical_im_folder + 'Para_2017/' 
        im_filenames = ['PA_S2_2017_B1_B2_B3_crop.tif', 
            'PA_S2_2017_B4_B5_B6_crop.tif', 
            'PA_S2_2017_B7_B8_B8A_crop.tif', 
            'PA_S2_2017_B9_B10_B11_crop.tif', 
            'PA_S2_2017_B12_crop.tif'] 
        path_cirrus_band = im_filenames[3]
        cirrus_band_id = 1

    if dataset_id == 'Para_2018': 
        dataset = Para() 
        path_optical_im = dataset.paths.optical_im_folder + 'Para_2018/' 
        im_filenames = ['COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif', 
            'COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif', 
            'COPERNICUS_S2_20180721_20180726_B7_B8_B8A.tif', 
            'COPERNICUS_S2_20180721_20180726_B9_B10_B11.tif', 
            'COPERNICUS_S2_20180721_20180726_B12.tif'] 
        path_cirrus_band = im_filenames[3]
        cirrus_band_id = 1

    elif dataset_id == 'Para_2019': 
        dataset = Para() 
        path_optical_im = dataset.paths.optical_im_folder + 'Para_2019/' 
        im_filenames = ['COPERNICUS_S2_20190721_20190726_B1_B2_B3.tif', 
            'COPERNICUS_S2_20190721_20190726_B4_B5_B6.tif', 
            'COPERNICUS_S2_20190721_20190726_B7_B8_B8A.tif', 
            'COPERNICUS_S2_20190721_20190726_B9_B10_B11.tif', 
            'COPERNICUS_S2_20190721_20190726_B12.tif'] 
        path_cirrus_band = im_filenames[3]
        cirrus_band_id = 1


    elif dataset_id == 'MT_2016': 
        dataset = MT() 
        path_optical_im = dataset.paths.optical_im_folder + 'MT_2016/' 
        im_filenames = ['MT_S2_2016_07_21_08_07_B1_B2_crop.tif', 
            'MT_S2_2016_07_21_08_07_B3_B4_crop.tif', 
            'MT_S2_2016_07_21_08_07_B5_B6_crop.tif', 
            'MT_S2_2016_07_21_08_07_B7_B8_crop.tif', 
            'MT_S2_2016_07_21_08_07_B8A_B9_crop.tif',
            'MT_S2_2016_07_21_08_07_B10_B11_crop.tif',
            'MT_S2_2016_07_21_08_07_B12_crop.tif'] 
        path_cirrus_band = im_filenames[5]
        cirrus_band_id = 0

    elif dataset_id == 'MT_2017': 
        dataset = MT() 
        path_optical_im = dataset.paths.optical_im_folder + 'MT_2017/' 
        im_filenames = ['MT_S2_07_26_28_2017_B1_B2_crop.tif', 
            'MT_S2_07_26_28_2017_B3_B4_crop.tif', 
            'MT_S2_07_26_28_2017_B5_B6_crop.tif', 
            'MT_S2_07_26_28_2017_B7_B8_crop.tif', 
            'MT_S2_07_26_28_2017_B8A_B9_crop.tif',
            'MT_S2_07_26_28_2017_B10_B11_crop.tif',
            'MT_S2_07_26_28_2017_B12_crop.tif'] 
        path_cirrus_band = im_filenames[5]
        cirrus_band_id = 0

    elif dataset_id == 'MT_2018': 
        dataset = MT() 
        path_optical_im = dataset.paths.optical_im_folder + 'MT_2018/' 
        im_filenames = ['MT_S2_07_26_28_31_2018_B1_B2_crop.tif', 
            'MT_S2_07_26_28_31_2018_B3_B4_crop.tif', 
            'MT_S2_07_26_28_31_2018_B5_B6_crop.tif', 
            'MT_S2_07_26_28_31_2018_B7_B8_crop.tif', 
            'MT_S2_07_26_28_31_2018_B8A_B9_crop.tif',
            'MT_S2_07_26_28_31_2018_B10_B11_crop.tif',
            'MT_S2_07_26_28_31_2018_B12_crop.tif'] 
        path_cirrus_band = im_filenames[5]
        cirrus_band_id = 0


    elif dataset_id == 'MT_2019': 
        dataset = MT() 
        path_optical_im = dataset.paths.optical_im_folder + 'MT_2019/' 
        im_filenames = ['S2_R1_MT_2019_08_02_2019_08_05_B1_B2.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B3_B4.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B5_B6.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B7_B8.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B8A_B9.tif',
            'S2_R1_MT_2019_08_02_2019_08_05_B10_B11.tif',
            'S2_R1_MT_2019_08_02_2019_08_05_B12.tif'] 
        path_cirrus_band = im_filenames[5]
        cirrus_band_id = 0

    elif dataset_id == 'MT_2020': 
        dataset = MT() 
        path_optical_im = dataset.paths.optical_im_folder + 'MT_2020/' 
        im_filenames = ['S2_R1_MT_2020_08_03_2020_08_15_B1_B2.tif', 
            'S2_R1_MT_2020_08_03_2020_08_15_B3_B4.tif', 
            'S2_R1_MT_2020_08_03_2020_08_15_B5_B6.tif', 
            'S2_R1_MT_2020_08_03_2020_08_15_B7_B8.tif', 
            'S2_R1_MT_2020_08_03_2020_08_15_B8A_B9.tif',
            'S2_R1_MT_2020_08_03_2020_08_15_B10_B11.tif',
            'S2_R1_MT_2020_08_03_2020_08_15_B12.tif'] 
        path_cirrus_band = im_filenames[5]
        cirrus_band_id = 0

    filename = dataset_id + '.npy' 
 
    optical_im = loadOpticalIm(path_optical_im, im_filenames) 
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
    cirrus = load_tiff_image(path_optical_im + path_cirrus_band)[cirrus_band_id]
    ic(np.min(cirrus), np.average(cirrus), np.max(cirrus))
    threshold = 19

    thin_cloud_mask = np.zeros_like(cirrus).astype(np.uint8)
    thin_cloud_mask[cirrus > threshold] = 1

    # === SAVE CLOUD MASK === #

    cloud_mask = np.zeros_like(thin_cloud_mask).astype(np.uint8)
    cloud_mask[thin_cloud_mask == 1] = 1
    cloud_mask[cloud_cloudshadow_mask == 1] = 1
    apply_shadow_mask = False
    if apply_shadow_mask == True:
        cloud_mask[cloud_cloudshadow_mask == -1] = 1

    plt.figure() 
    plt.imshow(cloud_mask) 
    plt.axis('off') 
    plt.savefig(path_optical_im + 'cloudmask_' + filename + '.png', dpi = 500)
    

    print("saving in... " + path_optical_im + "cloudmask_" + filename)
    np.save(path_optical_im + "cloudmask_" + filename, cloud_mask) 

