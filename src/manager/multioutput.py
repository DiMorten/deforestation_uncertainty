
from src.Logger import Logger
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
from src.manager.base import Manager
import src.loss
import src.uncertainty as uncertainty
import pathlib
import pdb
from src.evidential_learning import EvidentialLearning
import src.evidential_learning as evidential
import src.network as network
from tensorflow.keras.models import Model, load_model, Sequential
from src.evidential_learning import DirichletLayer
from numpy.core.numeric import Inf
class _EarlyStopping():
    def __init__(self, patience):
        self.patience = patience
        self.restartCounter()
    def restartCounter(self):
        self.counter = 0
    def increaseCounter(self):
        self.counter += 1
    def checkStopping(self):
        if self.counter >= self.patience:
            return True
        else:
            return False
        
class ManagerMultiOutput(Manager):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        super().__init__(config, dataset, patchesHandler, logger, grid_idx=grid_idx)
        self.network_architecture = network.build_resunet_dropout_spatial
        self.pred_entropy_single_idx = 0
        
    def train(self):

        metrics_all = []
            
        print('time: ', self.repetition_id)

        rows = self.patch_size
        cols = self.patch_size
        adam = Adam(lr = self.config['learning_rate'] , beta_1=0.9) # 1e-3
        
        loss = src.loss.weighted_categorical_crossentropy(self.weights)
        
        input_shape = (rows, cols, self.channels)
        self.model = self.network_architecture(input_shape, self.nb_filters, self.class_n)
        
        self.model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        self.model.summary()

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(self.path_models+ '/' + self.method +'_'+str(self.repetition_id)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
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

        # del self.train_gen_batch, self.valid_gen_batch

    def getMeanProb(self):
        self.mean_prob = np.mean(self.prob_rec, axis = -1)
        if self.classes_mode == True:
            self.mean_prob = self.mean_prob[...,1]            

    def preprocessProbRec(self):
        if self.classes_mode == False:
            self.prob_rec = np.transpose(self.prob_rec, (2, 0, 1))
            self.prob_rec = np.expand_dims(self.prob_rec, axis = -1)
        else:
            print(self.prob_rec.shape)
            self.prob_rec = np.transpose(self.prob_rec, (3, 0, 1, 2))

    def setUncertainty(self):

        if self.config['uncertainty_method'] == "pred_var":
            self.uncertainty_map = uncertainty.predictive_variance(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "MI":
            self.uncertainty_map = uncertainty.mutual_information(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "pred_entropy":
            self.uncertainty_map = uncertainty.predictive_entropy(self.prob_rec, self.classes_mode).astype(np.float32)

        elif self.config['uncertainty_method'] == "KL":
            self.uncertainty_map = uncertainty.expected_KL_divergence(self.prob_rec).astype(np.float32)

        elif self.config['uncertainty_method'] == "pred_entropy_single":
            self.uncertainty_map = uncertainty.single_experiment_entropy(
                self.prob_rec[self.pred_entropy_single_idx], self.classes_mode).astype(np.float32)

    def getPOIValues(self):
        self.snippet_poi_results = []

        lims_snippets = [self.dataset.previewLims1, self.dataset.previewLims2]
        for snippet_id, lims in enumerate(lims_snippets):
            for coord in self.dataset.snippet_coords["snippet_id{}".format(snippet_id)]:
                dict_ = {"snippet_id": snippet_id,
                        "coords": coord, # 10,1 alpha
                        "reference": self.label_mask[lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]]}
                
                predicted_coord = []
                for idx in range(self.prob_rec.shape[0]):
                    predicted_coord.append(self.prob_rec[idx][lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]])
                predicted_coord = np.array(predicted_coord)
                dict_["predicted"] = predicted_coord

                self.snippet_poi_results.append(dict_)

        return self.snippet_poi_results
class ManagerMCDropout(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = True
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.default_log_name = 'output/log/log_mcd.pkl'

class ManagerSingleRun(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = False
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.default_log_name = 'output/log/log_single_run.pkl'

class ManagerEvidential2(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = False
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.network_architecture = network.build_evidential_resunet
        self.weights = [0.1, 0.9]
        self.default_log_name = 'output/log/log_evidential.pkl'
        self.el = EvidentialLearning()


    def infer(self):

        self.h, self.w, self.c = self.image1_pad.shape
        self.c = self.channels
        patch_size_rows = self.h//self.n_rows
        patch_size_cols = self.w//self.n_cols
        num_patches_x = int(self.h/patch_size_rows)
        num_patches_y = int(self.w/patch_size_cols)

        ic(self.path_models+ '/' + self.method +'_'+str(self.repetition_id)+'.h5')
        model = load_model(self.path_models+ '/' + self.method +'_'+str(self.repetition_id)+'.h5', 
            compile=False, custom_objects={"DirichletLayer": DirichletLayer })
        if self.classes_mode == False:
            class_n = 3
        else:
            class_n = 2
        
        if self.config["loadInference"] == False:
            if self.config["save_probabilities"] == False:
                if self.classes_mode == False:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.config["inference_times"]), dtype = np.float32)
                else:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], class_n), dtype = np.float32)

                # self.prob_rec = np.zeros((image1_pad.shape[0],image1_pad.shape[1], class_n, self.config["inference_times"]), dtype = np.float32)
            print("Dropout training mode: {}".format(self.config['dropout_training']))
            new_model = self.network_architecture(input_shape=(patch_size_rows,patch_size_cols, self.c), 
                nb_filters = self.nb_filters, n_classes = class_n, dropout_seed = None, 
                training=self.config['dropout_training'], last_activation='relu')

            for l in range(1, len(model.layers)):
                new_model.layers[l].set_weights(model.layers[l].get_weights())
            
            self.patchesHandler.class_n = class_n

            with tf.device('/cpu:0'):
                tm = 0
                print('time: ', tm)


                # Recinstructing predicted map
                start_test = time.time()
                '''
                args_network = {'patch_size_rows': patch_size_rows,
                    'patch_size_cols': patch_size_cols,
                    'c': c,
                    'nb_filters': nb_filters,
                    'class_n': class_n,
                    'dropout_seed': inference_times}
                '''
                self.alpha_reconstructed = self.patchesHandler.infer(
                        new_model, self.image1_pad, self.h, self.w, 
                        num_patches_x, num_patches_y, patch_size_rows, 
                        patch_size_cols, classes_mode = self.classes_mode)
                prob_reconstructed, self.u_reconstructed = evidential.alpha_to_probability_and_uncertainty(
                    self.alpha_reconstructed)
                if self.classes_mode == False:
                    prob_reconstructed = prob_reconstructed[:,:,:,0:2]
                ts_time =  time.time() - start_test

                if self.config["save_probabilities"] == True:
                    np.save(self.path_maps+'/'+'prob_'+str(tm)+'.npy',prob_reconstructed) 
                else:
                    self.prob_rec = prob_reconstructed
                

                del prob_reconstructed
        del self.image1_pad

    def getMeanProb(self):
        self.mean_prob = self.prob_rec
      

    def applyProbabilityThreshold(self):
        print(self.mean_prob.shape)


        self.predicted_unpad = np.argmax(self.mean_prob, axis=-1).astype(np.int8)
        self.predicted_unpad[self.label_mask == 2] = 0

        
    def getValidationValuesForMetrics(self):
        self.label_mask_val = self.label_mask[self.mask_tr_val == 2]
        ic(self.label_mask_val.shape)

        self.mean_prob_val = self.mean_prob[...,1][self.mask_tr_val == 2]

        self.mean_prob_val = self.mean_prob_val[self.label_mask_val != 2]
        self.label_mask_val_valid = self.label_mask_val[self.label_mask_val != 2]
        ic(self.label_mask_val_valid.shape)

        self.predicted_val = self.predicted_unpad[self.mask_tr_val == 2]
        self.predicted_val = self.predicted_val[self.label_mask_val != 2]
    # to-do: pass to predictor. to do that, pass data to dataset class (dataset.image_stack, dataset.label, etc)

    def preprocessProbRec(self):
        pass

    def setUncertainty(self):
        self.uncertainty_map = self.u_reconstructed



    def getUncertaintyMetrics(self):
        predicted_thresholded = np.zeros_like(self.uncertainty).astype(np.int8)
        predicted_thresholded[self.uncertainty >= np.max(self.predicted_test,axis=-1)] = 1
        print(np.unique(predicted_thresholded, return_counts=True))

        predicted_test_classified_correct = self.predicted_test[
                predicted_thresholded == 0]
        label_current_deforestation_test_classified_correct = self.label_mask_current_deforestation_test[
                predicted_thresholded == 0]


        predicted_test_classified_incorrect = self.predicted_test[
                predicted_thresholded == 1]
        label_current_deforestation_test_classified_incorrect = self.label_mask_current_deforestation_test[
                predicted_thresholded == 1]

        uncertainty_classified_correct = self.uncertainty[
                predicted_thresholded == 0]
        uncertainty_classified_incorrect = self.uncertainty[
                predicted_thresholded == 1]
        print(np.min(uncertainty_classified_correct), np.mean(uncertainty_classified_correct), np.max(uncertainty_classified_correct))
        print(np.min(uncertainty_classified_incorrect), np.mean(uncertainty_classified_incorrect), np.max(uncertainty_classified_incorrect))

        print(label_current_deforestation_test_classified_correct.shape,
                predicted_test_classified_correct.shape)
        cm_correct = metrics.confusion_matrix(
                label_current_deforestation_test_classified_correct,
                predicted_test_classified_correct)
        print("cm_correct", cm_correct)

        TN_L = cm_correct[0,0]
        FN_L = cm_correct[1,0]
        TP_L = cm_correct[1,1]
        FP_L = cm_correct[0,1]

        ic(label_current_deforestation_test_classified_incorrect.shape,
                predicted_test_classified_incorrect.shape)

        cm_incorrect = metrics.confusion_matrix(
                label_current_deforestation_test_classified_incorrect,
                predicted_test_classified_incorrect)

        print("cm_incorrect", cm_incorrect)

        if cm_incorrect.shape[0] != 2: 
                ic(np.all(label_current_deforestation_test_classified_incorrect) == 0) 
                ic(np.all(predicted_test_classified_incorrect) == 0) 
                
                precision_L = np.nan 
                recall_L = np.nan 
                recall_Ltotal = np.nan 
                AA = len(label_current_deforestation_test_classified_incorrect) / len(self.label_mask_current_deforestation_test) 
                precision_H = np.nan 
                recall_H = np.nan 
        else:
                        
                TN_H = cm_incorrect[0,0]
                FN_H = cm_incorrect[1,0]
                TP_H = cm_incorrect[1,1]
                FP_H = cm_incorrect[0,1]
                
                precision_L = TP_L / (TP_L + FP_L)
                recall_L = TP_L / (TP_L + FN_L)
                
                precision_H = TP_H / (TP_H + FP_H)
                recall_H = TP_H / (TP_H + FN_H)
                
                recall_Ltotal = TP_L / (TP_L + FN_L + TP_H + FN_H)
                ic((TP_H + FN_H + FP_H + TN_H), len(self.label_mask_current_deforestation_test))
                AA = (TP_H + FN_H + FP_H + TN_H) / len(self.label_mask_current_deforestation_test)
                ic((TP_H + FN_H + FP_H + TN_H), len(self.label_mask_current_deforestation_test))


        self.m = {'precision_L': precision_L,
                'recall_L': recall_L,
                'recall_Ltotal': recall_Ltotal,
                'AA': AA,
                'precision_H': precision_H,
                'recall_H': recall_H}

        self.m['f1_L'] = 2*self.m['precision_L']*self.m['recall_L']/(self.m['precision_L']+self.m['recall_L'])
        self.m['f1_H'] = 2*self.m['precision_H']*self.m['recall_H']/(self.m['precision_H']+self.m['recall_H'])


    def train(self):

        metrics_all = []
            
        print('time: ', self.repetition_id)

        rows = self.patch_size
        cols = self.patch_size
        adam = Adam(lr = self.config['learning_rate'] , beta_1=0.9) # 1e-3
        
        # loss = src.loss.weighted_categorical_crossentropy(self.weights)
        # loss = self.el.categorical_crossentropy_envidential_learning
        # loss = self.el.weighted_categorical_crossentropy_evidential_learning(self.weights)
        loss = self.el.weighted_mse_loss(self.weights)

        input_shape = (rows, cols, self.channels)
        self.model = self.network_architecture(input_shape, self.nb_filters, self.class_n, last_activation='relu')
        # ,
        #                                        last_activation='relu'
        self.model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        self.model.summary()

        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
        checkpoint_filename = self.path_models+ '/' + self.method +'_'+str(self.repetition_id)+'.h5'
        checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
        callbacks_list = [earlystop, checkpoint]
        # train the model
        start_training = time.time()

        epochs = 500
        history_list = []

        val_loss = Inf
        es = _EarlyStopping(10)

        for epoch in range(1,epochs+1):
            print('Epoch:',epoch)
            print('Anneling Coeficient', self.el.an_)
            self.el.updateAnnealingCoeficient(epoch)
            history = self.model.fit(
                self.train_gen_batch, 
                epochs=1, 
                steps_per_epoch=self.len_X_train*3//self.train_gen.batch_size, # quitar el 3?
                validation_data=self.valid_gen_batch,
                validation_steps=self.len_X_valid*3//self.valid_gen.batch_size,
                callbacks=callbacks_list)
            history_list.append(history.history)
            new_val_loss = round(history.history['val_loss'][-1], 5)
            if new_val_loss < val_loss:
                # self.model.save(checkpoint_filename)
                val_loss = new_val_loss
                es.restartCounter()
                print("New best val loss. Val loss: {}. Early stop count: {}".format(
                    new_val_loss, es.counter))
            else:
                es.increaseCounter()
                print("Early stop count: {}".format(es.counter))
            if es.checkStopping() == True:
                print("Early stopping")
                print(es.counter, es.patience)
                print('Finished Training')
                break

        end_training = time.time() - start_training
        print('Training time: ', end_training)	


    def getOptimalUncertaintyThreshold(self, AA = 0.03, bounds = None):

        def getAAFromUncertaintyThreshold(threshold): 
            print(threshold)
            metrics_values2 = _metrics.getAA_Recall(self.uncertainty, 
                            self.label_mask_current_deforestation_test, 
                            self.predicted_test, [threshold])
            return np.abs(AA - metrics_values2[:,3].squeeze())
        if bounds is None:
            bounds = (np.min(self.uncertainty) + 0.0015, np.max(self.uncertainty)-0.0015)
        
        ic(bounds)
        minimum = optimize.minimize_scalar(getAAFromUncertaintyThreshold, 
            method='bounded', bounds=bounds, tol=0.0001)
        self.threshold_optimal = minimum.x
        ic(self.threshold_optimal)

# class ManagerEnsemble(ManagerMCDropout):
class ManagerEnsemble(ManagerMultiOutput):
    def __init__(self, config, dataset, patchesHandler, logger, grid_idx=0):
        config['dropout_training'] = False
        super().__init__(config, dataset, patchesHandler, logger, grid_idx)
        self.default_log_name = 'output/log/log_ensemble.pkl'

    def infer(self):
        self.h, self.w, self.c = self.image1_pad.shape
        self.c = self.channels
        patch_size_rows = self.h//self.n_rows
        patch_size_cols = self.w//self.n_cols
        num_patches_x = int(self.h/patch_size_rows)
        num_patches_y = int(self.w/patch_size_cols)

        if self.classes_mode == False:
            class_n = 3
            self.patchesHandler.class_n = class_n
        else:
            class_n = 2    
            self.patchesHandler.class_n = class_n + 1        
        if self.config["loadInference"] == False:
            if self.config["save_probabilities"] == False:
                if self.classes_mode == False:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.config["inference_times"]), dtype = np.float32)
                else:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], class_n, self.config["inference_times"]), dtype = np.float32)

            new_model = network.build_resunet_dropout_spatial(input_shape=(patch_size_rows,patch_size_cols, self.c), 
                nb_filters = self.nb_filters, n_classes = class_n, dropout_seed = None, training = False)
            t0 = time.time()

            # pathlib.Path(self.path_maps).mkdir(parents=True, exist_ok=True)
            with tf.device('/cpu:0'):
                for tm in range(0,self.config["inference_times"]):
                    print('time: ', tm)
                    
                    # Recinstructing predicted map
                    runtime_repetition_t0 = time.time()

                    path_exp = self.dataset.paths.experiment + 'exp' + str(self.exp) # exp_ids[tm]
                    path_models = path_exp + '/models'
                    # ic(path_models+ '/' + method +'_'+str(0)+'.h5')
                    path_repetition = path_models+ '/' + self.method +'_'+str(tm)+'.h5'
                    print("Loading model in:", path_repetition)
                    model = load_model(path_repetition, compile=False)
                    for l in range(1, len(model.layers)): #WHY 1?
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
                            
                    runtime_repetition =  time.time() - runtime_repetition_t0
                    print("runtime_repetition", round(runtime_repetition,2))
                    if self.config["save_probabilities"] == True:
                        
                        np.save(os.path.join(self.path_maps, 'prob_'+str(tm)+'.npy'),prob_reconstructed) 
                    else:
                        self.prob_rec[...,tm] = prob_reconstructed

                    del prob_reconstructed
            runtime = time.time() - t0
            print("Inference runtime", round(runtime,2))
            # print round time.time()


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

        
    def defineExperiment(self, exp_id):
        self.exp = exp_id

    def getPOIValues(self):
        self.snippet_poi_results = []

        lims_snippets = [self.dataset.previewLims1, self.dataset.previewLims2]
        for snippet_id, lims in enumerate(lims_snippets):
            for coord in self.dataset.snippet_coords["snippet_id{}".format(snippet_id)]:
                dict_ = {"snippet_id": snippet_id,
                        "coords": coord, # 10,1 alpha
                        "reference": self.label_mask[lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]]}
                
                predicted_coord = []
                for idx in range(self.prob_rec.shape[0]):
                    predicted_coord.append(self.prob_rec[idx][lims[0]:lims[1], lims[2]:lims[3]][coord[0], coord[1]])
                predicted_coord = np.array(predicted_coord)
                dict_["predicted"] = predicted_coord

                self.snippet_poi_results.append(dict_)

        return self.snippet_poi_results