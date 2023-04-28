
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
class TrainerMultiOutput(Trainer):
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
            adam = Adam(lr = self.config['learning_rate'] , beta_1=0.9) # 1e-3
            
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
        if self.classes_mode == True:
            self.mean_prob = self.mean_prob[...,1]            

    def preprocessProbRec(self):
        if self.classes_mode == False:
            self.prob_rec = np.transpose(self.prob_rec, (2, 0, 1))
            self.prob_rec = np.expand_dims(self.prob_rec, axis = -1)
        else:
            self.prob_rec = np.transpose(self.prob_rec, (3, 0, 1, 2))

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

class TrainerMCDropout(TrainerMultiOutput):
    pass

class TrainerEnsemble(TrainerMCDropout):

    def infer(self):
        self.h, self.w, self.c = self.image1_pad.shape
        self.c = self.channels
        patch_size_rows = self.h//self.n_rows
        patch_size_cols = self.w//self.n_cols
        num_patches_x = int(self.h/patch_size_rows)
        num_patches_y = int(self.w/patch_size_cols)


        class_n = 3
        self.classes_mode = True
        if self.config["loadInference"] == False:
            if self.config["save_probabilities"] == False:
                if self.classes_mode == False:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.config["inference_times"]), dtype = np.float32)
                else:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], class_n, self.config["inference_times"]), dtype = np.float32)

            new_model = utils_v1.build_resunet_dropout_spatial(input_shape=(patch_size_rows,patch_size_cols, self.c), 
                nb_filters = self.nb_filters, n_classes = class_n, dropout_seed = None, training = False)

            self.patchesHandler.class_n = class_n

            with tf.device('/cpu:0'):
                for tm in range(0,self.config["inference_times"]):
                    print('time: ', tm)
                    
                    # Recinstructing predicted map
                    start_test = time.time()

                    path_exp = self.dataset.paths.experiment + 'exp' + str(self.exp_ids[tm])
                    path_models = path_exp + '/models'
                    # ic(path_models+ '/' + method +'_'+str(0)+'.h5')
                    model = utils_v1.load_model(path_models+ '/' + self.method +'_'+str(0)+'.h5', compile=False)
                    for l in range(1, len(model.layers)):
                        new_model.layers[l].set_weights(model.layers[l].get_weights())
                    
                    '''
                    args_network = {'patch_size_rows': patch_size_rows,
                        'patch_size_cols': patch_size_cols,
                        'c': c,
                        'nb_filters': nb_filters,
                        'class_n': class_n,
                        'dropout_seed': inference_times}
                    '''
                    prob_reconstructed = self.patchesHandler.infer(
                            new_model, self.image1_pad, self.h, self.w, 
                            num_patches_x, num_patches_y, patch_size_rows, 
                            patch_size_cols, classes_mode = self.classes_mode)
                            
                    ts_time =  time.time() - start_test

                    if self.config["save_probabilities"] == True:
                        np.save(self.path_maps+'/'+'prob_'+str(tm)+'.npy',prob_reconstructed) 
                    else:
                        self.prob_rec[...,tm] = prob_reconstructed

                    del prob_reconstructed
        del self.image1_pad

    def run_predictor_repetition_single_entropy(self):
        # self.setExperimentPath()
        # self.createLogFolders()        
        self.setPadding()
        self.infer()
        self.loadPredictedProbabilities()
        self.getMeanProb()
        self.unpadMeanProb()
        self.squeezeLabel()
        self.setMeanProbNotConsideredAreas()
        self.getLabelTest()
        # self.getMAP()
        self.preprocessProbRec()
        # self.getUncertaintyToShow(self.pred_entropy)
        self.getLabelCurrentDeforestation()
        
        
        # min_metric = np.inf
        # max_metric = 0
        self.config['uncertainty_method'] = "pred_entropy_single"
        results = {}
        for idx in range(self.config['inference_times']):
            self.pred_entropy_single_idx = idx
            self.applyProbabilityThreshold() # start from here for single entropy loop
            self.getTestValues()
            self.removeSmallPolygons()
            self.calculateMetrics()
            self.getValidationValuesForMetrics()
            self.calculateMetricsValidation()
            calculateMAPWithoutSmallPolygons = False
            if calculateMAPWithoutSmallPolygons == True:
                self.calculateMAPWithoutSmallPolygons()
            self.getErrorMask()
            self.getErrorMaskToShowRGB()

            self.setUncertainty()
            self.getValidationValues2()
            self.getTestValues2()
            self.getOptimalUncertaintyThreshold()

            results["pred_entropy_single_{}".format(idx)] = self.getUncertaintyMetricsFromOptimalThreshold()
            
            '''
            results_tmp = self.getUncertaintyMetricsFromOptimalThreshold()
            metric = self.f1
            if metric > max_metric:
                max_metric = metric
                results["pred_entropy_single_max"] = results_tmp
            if metric < min_metric:
                min_metric = metric
                results["pred_entropy_single_min"] = results_tmp
            '''
        
        print("results", results)
        return results

        
    def defineExperiment(self, exp_ids):
        self.exp_ids = exp_ids
        self.exp = self.exp_ids[0]