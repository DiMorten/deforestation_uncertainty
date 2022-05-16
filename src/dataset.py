import sys
import numpy as np
from icecream import ic
import pdb
import scipy
sys.path.append("..")
from src.paths import PathsPara
import utils_v1
class Dataset():
    def getTrainValTestMasks(self, mask_tiles):
        tiles_tr, tiles_val, tiles_ts = self.calculateTiles()

        print('Training tiles: ', tiles_tr)
        print('Validation tiles: ', tiles_val)
        print('Test tiles: ', tiles_ts)

        # Training and validation mask
        mask_tr_val = np.zeros((mask_tiles.shape)).astype('uint8')

        for tr_ in tiles_tr:
            mask_tr_val[mask_tiles == tr_] = 1

        for val_ in tiles_val:
            mask_tr_val[mask_tiles == val_] = 2

        mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('uint8')
        for ts_ in tiles_ts:
            mask_amazon_ts[mask_tiles == ts_] = 1
        return mask_tr_val, mask_amazon_ts        
    def getLabelCurrentDeforestation(self, label_mask, selected_class = 1):
        label_mask_current_deforestation = label_mask.copy()
        label_mask_current_deforestation[label_mask_current_deforestation == selected_class] = 10
        label_mask_current_deforestation[label_mask_current_deforestation != 10] = 0
        label_mask_current_deforestation[label_mask_current_deforestation == 10] = 1
        return label_mask_current_deforestation.astype(np.uint8)
    def createDistMap(self, past_ref, th_sup = 800):
        print('Generating distance map ... ')
        # import scipy

        dist_past = past_ref.copy()
        dist_matrix = scipy.ndimage.distance_transform_edt(dist_past == 0)
        dist_matrix[dist_matrix >= th_sup] = th_sup
        dist_norm = (dist_matrix-np.min(dist_matrix))/(np.max(dist_matrix)-np.min(dist_matrix))
        return dist_norm        
class Para(Dataset):
    def __init__(self):
        self.paths = PathsPara()

        self.grid_x, self.grid_y = 5,4

        self.label_filename = 'mask_label_17730x9203.npy'
    def loadLabel(self):
        label = np.load(self.paths.label + self.label_filename).astype('uint8')
        return label
    def loadInputImage(self):
        return np.load(self.paths.optical_im + 'optical_im.npy').astype('float32')
    def calculateTiles(self):
        # Defining tiles for training, validation and test sets
        tiles_tr = [1,3,5,8,11,13,14,20] 
        tiles_val = [6,19]
        tiles_ts = list(set(np.arange(self.grid_x * self.grid_y)+1)-set(tiles_tr)-set(tiles_val))
        return tiles_tr, tiles_val, tiles_ts
    def loadPastDeforestationLabel(self):
        label_past_deforestation = self.loadLabel()

        label_past_deforestation[label_past_deforestation == 1] = 0
        label_past_deforestation[label_past_deforestation == 2] = 1

        return label_past_deforestation

class ParaDistanceMap(Para):
    def loadInputImage(self):
        image_stack = super().loadInputImage()
        image_stack = self.addNpyBandToInput(image_stack, 
                self.paths.distance_map_past_deforestation)
        image_stack = self.addNpyBandToInput(image_stack, 
                self.paths.distance_map_past_deforestation_2018)
        image_stack = self.addNpyBandToInput(image_stack, 
                self.paths.distance_map_past_deforestation_2017)
        image_stack = self.addNpyBandToInput(image_stack, 
                self.paths.distance_map_past_deforestation_2016)

        ic(image_stack.shape)
        return image_stack  
    def addNpyBandToInput(self, image_stack, path):
        band = np.load(path).astype(np.float32)
        band = np.expand_dims(band, axis = -1)
        image_stack = np.concatenate((band, image_stack), axis = -1)  
        return image_stack    

class ParaDeforestationTime(Para):
    def loadInputImage(self):
        image_stack = super().loadInputImage()
        image_stack = self.addDeforestationTime(image_stack)
        # image_stack = self.addPastDeforestation(image_stack)
        ic(image_stack.shape)
        return image_stack  
    def addDeforestationTime(self, image_stack):
        deforestation_time = np.load(self.paths.label + 'deforestation_time_normalized_2018_2019.npy') # has past deforestation up to 2018
        # 2018 = 0, 2017 = 1, 2016 = 2,....
        ic(deforestation_time.dtype, deforestation_time.shape)
        self.usePastDeforestationWithoutDistance = False
        if self.usePastDeforestationWithoutDistance == True:
            ic(np.unique(deforestation_time, return_counts=True))
            deforestation_time[deforestation_time>0] = 1
        ic(np.unique(deforestation_time, return_counts=True))

        
        ic(deforestation_time.shape, image_stack.shape)
        image_stack = np.concatenate((deforestation_time, image_stack), axis = -1)
        del deforestation_time  
        return image_stack
    '''
    def addPastDeforestation(self, image_stack):
        past_deforestation = self.loadPastDeforestationLabel().astype(np.uint8)
        ic(past_deforestation.shape)
        # pdb.set_trace()
        past_deforestation = past_deforestation[..., np.newaxis]
        # pdb.set_trace()
        image_stack = np.concatenate((past_deforestation, image_stack), axis = -1)
        del past_deforestation  
        return image_stack    
    '''    
