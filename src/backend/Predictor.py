
from src.backend.Logger import Logger
import utils_v1
from icecream import ic
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from src.patchesHandler import PatchesHandler, PatchesHandlerMultipleDates, PatchesHandlerEvidential
import time
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

class Predictor():
    def __init__(self, dataset, patchesHandler):
        self.dataset = dataset
        self.patch_size = 128
        self.overlap = 0.7
        self.batch_size = 32
        self.class_n = 3
        self.logger = Logger()
        self.patchesHandler = patchesHandler


        self.times = 1
        self.method = 'resunet'
        self.nb_filters = [16, 32, 64, 128, 256]
        self.weights = [0.1, 0.9, 0]

        self.n_pool = 3
        self.n_rows = 5
        self.n_cols = 4



        #%% Test loop

        self.metrics_ts = []
        rows, cols = self.image_stack.shape[:2]
        pad_rows = rows - np.ceil(rows/(self.n_rows*2**self.n_pool))*self.n_rows*2**self.n_pool
        pad_cols = cols - np.ceil(cols/(self.n_cols*2**self.n_pool))*self.n_cols*2**self.n_pool
        print(pad_rows, pad_cols)

        npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
        image1_pad = np.pad(self.image_stack, pad_width=npad, mode='reflect')
        # del image_stack
        class_n = 3    