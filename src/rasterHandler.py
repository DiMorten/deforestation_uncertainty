
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
def padForGeorreferencingChannels(im):
    pad_value = -1
    print(im.shape)
    im_pad = np.pad(im, ((0,3), (0,0), (0,0)), constant_values = pad_value)
    print(im_pad.shape)
    im_pad = np.pad(im_pad, ((0,4000), (3000,0), (0,0)), constant_values = pad_value)
    print(im_pad.shape)
    return im_pad
