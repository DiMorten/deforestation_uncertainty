
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
from sklearn import metrics
from sklearn.metrics import f1_score
from src import metrics as _metrics
import cv2
from enum import Enum
import matplotlib.pyplot as plt
from scipy import optimize  
from src.trainer.base import Trainer
import src.loss
import src.uncertainty as uncertainty

class TrainerMCDropout(Trainer):
    def __init__(self, config, dataset, patchesHandler, grid_idx=0):
        super().__init__(config, dataset, patchesHandler, grid_idx=grid_idx)
        self.network_architecture = utils_v1.build_resunet_dropout_spatial
        self.pred_entropy_single_idx = 0
    def train(self):

        metrics_all = []
        # if self.training == True:
        for tm in range(0,self.times):
            print('time: ', tm)

            rows = self.patch_size
            cols = self.patch_size
            adam = Adam(lr = 1e-3 , beta_1=0.9)
            
            loss = src.loss.weighted_categorical_crossentropy(self.weights)
            
            input_shape = (rows, cols, self.channels)
            self.model = self.network_architecture(input_shape, self.nb_filters, self.class_n)
            
            self.model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
            self.model.summary()

            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
            checkpoint = ModelCheckpoint(self.path_models+ '/' + self.method +'_'+str(tm)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
            callbacks_list = [earlystop, checkpoint]
            # train the model
            start_training = time.time()
            self.history = self.model.fit_generator(self.train_gen_batch,
                                    steps_per_epoch=self.len_X_train*3//self.train_gen.batch_size,
                                    validation_data=self.valid_gen_batch,
                                    validation_steps=self.len_X_valid*3//self.valid_gen.batch_size,
                                    epochs=100,
                                    callbacks=callbacks_list)
            end_training = time.time() - start_training
            # metrics_all.append(end_training)

        # Saving training time
        # np.save(path_exp+'/metrics_tr.npy', metrics_all)
        del self.train_gen_batch, self.valid_gen_batch

    def getMeanProb(self):
        self.mean_prob = np.mean(self.prob_rec, axis = -1)
    def preprocessProbRec(self):
        self.prob_rec = np.transpose(self.prob_rec, (2, 0, 1))
        self.prob_rec = np.expand_dims(self.prob_rec, axis = -1)


    def setUncertainty(self):

        if self.config['uncertainty_method'] == "pred_var":
            self.uncertainty_map = uncertainty.predictive_variance(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "MI":
            self.uncertainty_map = uncertainty.mutual_information(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "pred_entropy":
            self.uncertainty_map = uncertainty.predictive_entropy(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "KL":
            self.uncertainty_map = uncertainty.expected_KL_divergence(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "pred_entropy_single":
            self.uncertainty_map = uncertainty.single_experiment_entropy(
                self.prob_rec[self.pred_entropy_single_idx]).astype(np.float32)
        