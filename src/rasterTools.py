
import os
import numpy as np 
import utils_v1
from icecream import ic
from osgeo import gdal
import pdb
from sklearn.preprocessing._data import _handle_zeros_in_scale
import cv2
from matplotlib import pyplot as plt

def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    im = gdal_header.ReadAsArray()
    return im

def scaleIm(im, scale):
    im = np.squeeze(im)
    if scale != 1:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return im

def loadOpticalIm(path_optical_im, im_filenames, scale_list = None): 
    band_count = 0 
 
    for i, im_filename in enumerate(im_filenames): 
        ic(os.path.join(path_optical_im, im_filename))         
        band = load_tiff_image(os.path.join(path_optical_im, im_filename)).astype('float32') 
        ic(band.shape) 
        if len(band.shape) == 2: band = band[np.newaxis, ...] 
        if scale_list != None:
            band = np.expand_dims(scaleIm(band, scale_list[i]), axis=0)
        if i: 
            ic(band.shape, optical_im.shape) 
            optical_im = np.concatenate((optical_im, band), axis=0) 
        else: 
            optical_im = band 
    del band  
    return optical_im     


import rasterio
import numpy as np
from icecream import ic
# rasterio.uint16
def GeoReference_Raster_from_Source_data(source_file, 
    numpy_image, target_file, bands, nodata=0, dtype = rasterio.float32):

    with rasterio.open(source_file) as src:
        ras_meta = src.profile

    ras_meta.update(count=bands)
    ras_meta.update(dtype=dtype)
    ras_meta.update(nodata=nodata)

    with rasterio.open(target_file, 'w', **ras_meta) as dst:
        dst.write(numpy_image)

def padForGeorreferencing(im, pad_value = -1):
    
    ic(im.shape)
    im_pad = np.pad(im, ((0,3), (0,0)), constant_values = pad_value)
    ic(im_pad.shape)
    im_pad = np.pad(im_pad, ((0,4000), (3000,0)), constant_values = pad_value)
    ic(im_pad.shape)
    return im_pad
def padForGeorreferencingPA(im, pad_value = -1):
    
    ic(im.shape)
    im_pad = np.pad(im, ((0,0), (0,3)), constant_values = pad_value)

    ic(im_pad.shape)
    return im_pad
def padForGeorreferencingChannels(im):
    pad_value = -1
    print(im.shape)
    im_pad = np.pad(im, ((0,3), (0,0), (0,0)), constant_values = pad_value)
    print(im_pad.shape)
    im_pad = np.pad(im_pad, ((0,4000), (3000,0), (0,0)), constant_values = pad_value)
    print(im_pad.shape)
    return im_pad
def save_georreferenced(im, original_im_path, produced_im_path, bands = 1, plot = False):
    if len(im.shape) == 2:
        pad_values = ((0, 0), (0, 1))
    else:
        pad_values = ((0, 0), (0, 1), (0, 0))
    im_pad = np.pad(im, pad_values)
    print("im_pad.shape", im_pad.shape)
    if plot == True:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im_pad, cmap='jet')
        plt.axis('off')

    if len(im_pad.shape) == 2:
        im_pad = np.expand_dims(im_pad, axis=0)
    else:
        im_pad = np.transpose(im_pad, (2, 0, 1))
    print("im_pad.shape", im_pad.shape)
    GeoReference_Raster_from_Source_data(original_im_path, 
                    im_pad.astype(np.float32), produced_im_path, bands = bands,
                    nodata=-1)
    del im_pad
