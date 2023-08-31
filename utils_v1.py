# Import libraries
import os
import sys
import time
import math
import random
import numpy as np
from PIL import Image
import tensorflow as tf
#from libtiff import TIFF
import skimage.morphology 
from osgeo import ogr, gdal
from scipy import ndimage
#import tifffile
import matplotlib.pyplot as plt
from skimage.filters import rank
from sklearn.utils import shuffle
from skimage.morphology import disk
from skimage.transform import resize
import tensorflow.keras.backend as K
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix
from skimage.util.shape import view_as_windows
from sklearn.metrics import average_precision_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)


# Functions
def load_optical_image(patch):
    # Read tiff Image
    print (patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    #img_tif = TIFF.open(patch)
    #img = img_tif.read_image()
    img = np.transpose(img.copy(), (1, 2, 0))
    print('Image shape :', img.shape)
    return img

def load_SAR_image(patch):
    #Function to read SAR images
    print (patch)
    gdal_header = gdal.Open(patch)
    db_img = gdal_header.ReadAsArray()
    #img_tif = TIFF.open(patch)
    #db_img = img_tif.read_image()
    #db_img = np.transpose(db_img, (1, 2, 0))
    temp_db_img = 10**(db_img/10)
    temp_db_img[temp_db_img>1] = 1
    return temp_db_img

def load_tiff_image(patch):
    # Read tiff Image
    print (patch)
    gdal_header = gdal.Open(patch)
    img = gdal_header.ReadAsArray()
    return img

def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
    return img


def create_mask(size_rows, size_cols, grid_size=(6,3)):
    num_tiles_rows = size_rows//grid_size[0]
    num_tiles_cols = size_cols//grid_size[1]
    print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows*grid_size[0], num_tiles_cols*grid_size[1]), 
        dtype=np.uint8)
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count+1
            mask[num_tiles_rows*i:(num_tiles_rows*i+num_tiles_rows), num_tiles_cols*j:(num_tiles_cols*j+num_tiles_cols)] = patch*count
    #plt.imshow(mask)
    print('Mask size: ', mask.shape)
    return mask

def create_idx_image(ref_mask):
    im_idx = np.arange(ref_mask.shape[0] * ref_mask.shape[1]).reshape(ref_mask.shape[0] , ref_mask.shape[1])
    return im_idx

def extract_patches(im_idx, patch_size, overlap):
    '''overlap range: 0 - 1 '''
    row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
    patches = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))
    return patches

def retrieve_idx_percentage(reference, patches_idx_set, patch_size, pertentage = 5):
    count = 0
    new_idx_patches = []
    reference_vec = reference.reshape(reference.shape[0]*reference.shape[1])
    for patchs_idx in patches_idx_set:
        patch_ref = reference_vec[patchs_idx]
        class1 = patch_ref[patch_ref==1]
        if len(class1) >= int((patch_size**2)*(pertentage/100)):
            count = count + 1
            new_idx_patches.append(patchs_idx)
    return np.asarray(new_idx_patches)

def extract_patches_mask_indices(input_image, patch_size, stride):
    h, w = input_image.shape
    image_indices = np.arange(h*w).reshape(h,w)
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image_indices, window_shape_array, step = stride))    
    num_row,num_col,row,col = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col)
    return patches_array


def normalization(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1

def get_patches_batch(image, rows, cols, radio, batch):
    temp = []
    for i in range(0, batch):
        batch_patches = image[rows[i]-radio:rows[i]+radio+1, cols[i]-radio:cols[i]+radio+1, :]
        temp.append(batch_patches)
    patches = np.asarray(temp)
    return patches

def pred_recostruction(patch_size, pred_labels, image_ref):
    # Reconstruction 
    stride = patch_size
    h, w = image_ref.shape
    num_patches_h = int(h/stride)
    num_patches_w = int(w/stride)
    count = 0
    img_reconstructed = np.zeros((num_patches_h*stride,num_patches_w*stride))
    for i in range(0,num_patches_w):
        for j in range(0,num_patches_h):
            img_reconstructed[stride*j:stride*(j+1),stride*i:stride*(i+1)]=pred_labels[count]
            #img_reconstructed[32*i:32*(i+1),32*j:32*(j+1)]=p_labels[count]
            count+=1
    return img_reconstructed



def mask_no_considered(image_ref, past_ref, buffer):
    # Creation of buffer for pixel no considered
    image_ref_ = image_ref.copy()
    im_dilate = skimage.morphology.dilation(image_ref_, disk(buffer))
    im_erosion = skimage.morphology.erosion(image_ref_, disk(buffer))
    inner_buffer = image_ref_ - im_erosion
    inner_buffer[inner_buffer == 1] = 2
    outer_buffer = im_dilate-image_ref_
    outer_buffer[outer_buffer == 1] = 2
    
    # 1 deforestation, 2 unknown
    image_ref_[outer_buffer + inner_buffer == 2 ] = 2
    #image_ref_[outer_buffer == 2 ] = 2
    image_ref_[past_ref == 1] = 2
    return image_ref_


def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    thresholds = thresholds_    
    metrics_all = []
    
    for thr in thresholds:
        print(thr)  

        img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
        img_reconstructed[prob_map >= thr] = 1
    
        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0
        
        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        #ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed==2] = 0
        #mask_borders[ref_reconstructed==-1] = 0
        
        mask_no_consider = mask_areas_pred * mask_borders 
        ref_consider = mask_no_consider * ref_reconstructed
        pred_consider = mask_no_consider*img_reconstructed
        
        ref_final = ref_consider[mask_amazon_ts_==1]
        pre_final = pred_consider[mask_amazon_ts_==1]
        
        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        mm = np.hstack((recall_, precision_))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_


def excludeBackgroundAreasFromTest(vector_test, label_mask_test):
    return vector_test[label_mask_test != 2]

def getTestVectorFromIm(im, mask, mask_return_value = 1):
    return im[mask == mask_return_value]

def unpadIm(im, npad):
    return im[:-npad[0][1], :-npad[1][1]]

import tracemalloc
from collections import Counter
import linecache

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
