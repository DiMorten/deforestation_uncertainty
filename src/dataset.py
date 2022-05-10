import sys
import numpy as np

sys.path.append("..")
from src.paths import PathsPara

class Dataset():
    def getTrainValTestMasks(self, mask_tiles):
        tiles_tr, tiles_val, tiles_ts = self.calculateTiles()

        print('Training tiles: ', tiles_tr)
        print('Validation tiles: ', tiles_val)
        print('Test tiles: ', tiles_ts)

        # Training and validation mask
        mask_tr_val = np.zeros((mask_tiles.shape)).astype('float32')

        for tr_ in tiles_tr:
            mask_tr_val[mask_tiles == tr_] = 1

        for val_ in tiles_val:
            mask_tr_val[mask_tiles == val_] = 2

        mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('float32')
        for ts_ in tiles_ts:
            mask_amazon_ts[mask_tiles == ts_] = 1
        return mask_tr_val, mask_amazon_ts        
    def getLabelCurrentDeforestation(self, label_mask, selected_class = 1):
        label_mask_current_deforestation = label_mask.copy()
        label_mask_current_deforestation[label_mask_current_deforestation == selected_class] = 10
        label_mask_current_deforestation[label_mask_current_deforestation != 10] = 0
        label_mask_current_deforestation[label_mask_current_deforestation == 10] = 1
        return label_mask_current_deforestation
class Para(Dataset):
    def __init__(self):
        self.paths = PathsPara()

        self.grid_x, self.grid_y = 5,4

        self.label_filename = 'mask_label_17730x9203.npy'
    def loadLabel(self):
        label = np.load(self.paths.label + self.label_filename).astype('float32')
        return label
    def loadInputImage(self):
        return np.load(self.paths.optical_im + 'optical_im.npy').astype('float32')
    def calculateTiles(self):
        # Defining tiles for training, validation and test sets
        tiles_tr = [1,3,5,8,11,13,14,20] 
        tiles_val = [6,19]
        tiles_ts = list(set(np.arange(self.grid_x * self.grid_y)+1)-set(tiles_tr)-set(tiles_val))
        return tiles_tr, tiles_val, tiles_ts


class ParaDeforestationTime(Para):
    def loadInputImage(self):
        image_stack = super().loadInputImage()
        image_stack = self.addDeforestationTime(image_stack)
        ic(image_stack.shape)
        return image_stack  
    def addDeforestationTime(self, image_stack):
        deforestation_time = np.load(self.paths.label + 'deforestation_time_normalized.npy')
        image_stack = np.concatenate((deforestation_time, image_stack), axis = -1)  
        return image_stack
        
