
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

class Trainer():
    def __init__(self, config, dataset, patchesHandler, grid_idx=0):
        self.config = config
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
        # self.weights = [0.0025, 0.9975, 0]
        self.title_name = 'ResUnet'


        self.grid_idx = grid_idx

        self.network = None
        self.UncertaintyMethod = Enum('UncertaintyMethod', 'pred_var MI pred_entropy KL pred_entropy_single')

    def defineExperiment(self, exp):
        self.exp = exp

    def setExperimentPath(self):
        self.path_exp = self.dataset.paths.experiment + 'exp' + str(self.exp)
        self.path_models = self.path_exp+'/models'
        self.path_maps = self.path_exp+'/pred_maps'
        
    def createLogFolders(self):
        self.logger.createLogFolders(self.dataset)

        # Creating folder for the experiment

        if not os.path.exists(self.path_exp):
            os.makedirs(self.path_exp)   
        if not os.path.exists(self.path_models):
            os.makedirs(self.path_models)   
        if not os.path.exists(self.path_maps):
            os.makedirs(self.path_maps)

    def loadLabel(self):
        self.label_mask = self.dataset.loadLabel()
        print('Mask label shape: ', '\n', self.label_mask.shape, '\n', 'Unique values: ', '\n', np.unique(self.label_mask))


    def createTrainValTestTiles(self):
        self.mask_tiles = utils_v1.create_mask(self.label_mask.shape[0], self.label_mask.shape[1], 
                grid_size=(self.dataset.grid_x, self.dataset.grid_y))
        self.label_mask = self.label_mask[:self.mask_tiles.shape[0], :self.mask_tiles.shape[1]]

    def getLabelCurrentDeforestation(self):
        self.label_mask_current_deforestation = self.dataset.getLabelCurrentDeforestation(
            self.label_mask)

    def loadInputImage(self):
        # Loading image stack
        self.image_stack = self.dataset.loadInputImage()

        print('Image shape: ', self.image_stack.shape)
        self.channels = self.image_stack.shape[-1]
        self.image_stack = self.image_stack[:self.mask_tiles.shape[0], :self.mask_tiles.shape[1],:]
        print('mask: ',self.mask_tiles.shape)
        print('image stack: ', self.image_stack.shape)
        print('ref :', self.label_mask.shape)
        #plt.imshow(self.mask_tiles)

    def getImageChannels(self):
        ic(self.image_stack.shape)
        self.channels = self.image_stack.shape[-1]
        ic(self.channels)

    def getTrainValTestMasks(self):
        self.mask_tr_val, self.mask_amazon_ts = self.dataset.getTrainValTestMasks(self.mask_tiles)

    def createIdxImage(self):
        self.im_idx = self.patchesHandler.create_idx_image(self.label_mask)
    
    def extractCoords(self):
        self.coords = self.patchesHandler.extract_patches(
            self.im_idx, patch_size=(self.patch_size, self.patch_size, 2), 
            overlap=self.overlap)
            
    def trainTestSplit(self):
        self.coords_train, self.coords_val = self.patchesHandler.trainTestSplit(self.coords,
		    self.mask_tr_val, patch_size=(self.patch_size, self.patch_size, 2))        
        ic(self.coords_train.shape, self.coords_val.shape)            

    def retrieveSamplesOfInterest(self, percentage = 0.2):
        # Keeping patches with 2% of def class
        self.coords_train = self.patchesHandler.retrieve_idx_percentage(self.label_mask, self.coords_train, 
                self.patch_size, pertentage = percentage)
        self.coords_val = self.patchesHandler.retrieve_idx_percentage(self.label_mask, self.coords_val, 
                self.patch_size, pertentage = percentage)
        print('training samples: ', self.coords_train.shape, 
                'validation samples: ', self.coords_val.shape)


    def getGenerators(self):
        train_datagen = ImageDataGenerator()
        valid_datagen = ImageDataGenerator()
        # pdb.set_trace()
        self.len_X_train = self.coords_train.shape[0]
        self.len_X_valid = self.coords_val.shape[0]

        self.train_gen = train_datagen.flow(
                np.expand_dims(np.expand_dims(self.coords_train, axis = -1), axis = -1), 
                np.expand_dims(np.expand_dims(self.coords_train, axis = -1), axis = -1),
                batch_size=self.batch_size,
                shuffle=True)
        # pdb.set_trace()

        self.valid_gen = valid_datagen.flow(
                np.expand_dims(np.expand_dims(self.coords_val, axis = -1), axis = -1), 
                np.expand_dims(np.expand_dims(self.coords_val, axis = -1), axis = -1),
                batch_size=self.batch_size,
                shuffle=False)

        
        self.train_gen_batch = self.patchesHandler.batch_generator(self.train_gen,
                self.image_stack, self.label_mask, self.patch_size, self.class_n)
        self.valid_gen_batch = self.patchesHandler.batch_generator(self.valid_gen,
                self.image_stack, self.label_mask, self.patch_size, self.class_n)

    def fixChannelNumber(self):
        if type(self.patchesHandler) == PatchesHandlerMultipleDates:
            self.channels = self.patchesHandler.input_image_shape  

    # def train(self):
    #     pass

    def loadDataset(self):
        self.loadLabel()
        self.createTrainValTestTiles()
        self.getLabelCurrentDeforestation()
        self.loadInputImage()
        self.getTrainValTestMasks()

    def run(self):
        self.createIdxImage()
        self.extractCoords()
        self.trainTestSplit()
        self.retrieveSamplesOfInterest()
        self.getGenerators()
        self.fixChannelNumber()
        self.train()

    def snipDataset(self, idx=0):
        self.logger.snipDataset(idx, self.coords_train, self.patchesHandler, 
            self.image_stack, self.label_mask)


    def plotHistory(self):
        self.logger.plotHistory(self.history)


    # to-do: pass to predictor. to do that, pass data to dataset class (dataset.image_stack, dataset.label, etc)

    def run_predictor(self):
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
        self.applyProbabilityThreshold()
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
        # self.getUncertaintyAAValues()
        # trainer.getUncertaintyAAAuditedValues()
        self.getOptimalUncertaintyThreshold()
        result = self.getUncertaintyMetricsFromOptimalThreshold()
        return result
    def setPadding(self):
        self.metrics_ts = []
        self.n_pool = 3
        self.n_rows = 5
        self.n_cols = 4
        self.rows, self.cols = self.image_stack.shape[:2]
        pad_rows = self.rows - np.ceil(self.rows/(self.n_rows*2**self.n_pool))*self.n_rows*2**self.n_pool
        pad_cols = self.cols - np.ceil(self.cols/(self.n_cols*2**self.n_pool))*self.n_cols*2**self.n_pool
        print(pad_rows, pad_cols)

        self.npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
        self.image1_pad = np.pad(self.image_stack, pad_width=self.npad, mode='reflect')
        # del image_stack
        self.class_n = 3
    def infer(self):

        self.h, self.w, self.c = self.image1_pad.shape
        self.c = self.channels
        patch_size_rows = self.h//self.n_rows
        patch_size_cols = self.w//self.n_cols
        num_patches_x = int(self.h/patch_size_rows)
        num_patches_y = int(self.w/patch_size_cols)

        ic(self.path_models+ '/' + self.method +'_'+str(0)+'.h5')
        model = utils_v1.load_model(self.path_models+ '/' + self.method +'_'+str(0)+'.h5', compile=False)
        class_n = 3
        self.classes_mode = True
        if self.config["loadInference"] == False:
            if self.config["save_probabilities"] == False:
                if self.classes_mode == False:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.config["inference_times"]), dtype = np.float32)
                else:
                    self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], class_n, self.config["inference_times"]), dtype = np.float32)

                # self.prob_rec = np.zeros((image1_pad.shape[0],image1_pad.shape[1], class_n, self.config["inference_times"]), dtype = np.float32)

            new_model = utils_v1.build_resunet_dropout_spatial(input_shape=(patch_size_rows,patch_size_cols, self.c), 
                nb_filters = self.nb_filters, n_classes = class_n, dropout_seed = None)

            for l in range(1, len(model.layers)):
                new_model.layers[l].set_weights(model.layers[l].get_weights())
            
            self.patchesHandler.class_n = class_n

            metrics_all =[]
            with tf.device('/cpu:0'):
                for tm in range(0,self.config["inference_times"]):

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
                    prob_reconstructed = self.patchesHandler.infer(
                            new_model, self.image1_pad, self.h, self.w, 
                            num_patches_x, num_patches_y, patch_size_rows, 
                            patch_size_cols, classes_mode = self.classes_mode)
                            
                    ts_time =  time.time() - start_test

                    if self.config["save_probabilities"] == True:
                        np.save(self.path_maps+'/'+'prob_'+str(tm)+'.npy',prob_reconstructed) 
                    else:
                        self.prob_rec[...,tm] = prob_reconstructed
                    
                    metrics_all.append(ts_time)
                    del prob_reconstructed
                metrics_ = np.asarray(metrics_all)
                # Saving test time
                np.save(self.path_exp+'/metrics_ts.npy', metrics_)
        del self.image1_pad
    def loadPredictedProbabilities(self):
        if self.config["save_probabilities"] == True:
            self.prob_rec = np.zeros((self.h, self.w, self.config["inference_times"]), dtype = np.float32)

            for tm in range(0, self.config["inference_times"]):
                print(tm)
                self.prob_rec[:,:,tm] = np.load(self.path_maps+'/'+'prob_'+str(tm)+'.npy').astype(np.float32)

    def getMeanProb():
        pass

    def unpadMeanProb(self):   
        self.mean_prob = self.mean_prob[:self.label_mask.shape[0], :self.label_mask.shape[1]]        
    def squeezeLabel(self):
        self.label_mask = np.squeeze(self.label_mask)

    def setMeanProbNotConsideredAreas(self):
        self.mean_prob = self.mean_prob.copy()
        self.mean_prob[self.label_mask == 2] = 0

    def getLabelTest(self):
        self.label_test = self.label_mask[self.mask_amazon_ts == 1]
        self.mean_prob_test = self.mean_prob[self.mask_amazon_ts == 1]
        self.mean_prob_test = self.mean_prob_test[self.label_test != 2]
        self.label_test = self.label_test[self.label_test != 2]

        print(self.label_test.shape)
        print(np.unique(self.label_test, return_counts=True))


    def getMAP(self):
        mAP = round(metrics.average_precision_score(self.label_test, 
                self.mean_prob_test)*100, 2)
        print(mAP)

    def preprocessProbRec(self):
        pass
    def getUncertaintyToShow(self):
        
        self.uncertainty_to_show = self.uncertainty_map.copy()[:self.label_mask.shape[0], :self.label_mask.shape[1]]

        self.uncertainty_to_show[self.label_mask == 2] = 0

    def getLabelCurrentDeforestation(self):
        self.label_mask_current_deforestation = self.label_mask.copy()
        self.label_mask_current_deforestation[self.label_mask_current_deforestation==2] = 0


    def applyProbabilityThreshold(self):
        print(self.mean_prob.shape)
        self.predicted = np.zeros_like(self.mean_prob)
        self.threshold = 0.5

        if self.config['uncertainty_method'] != "pred_entropy_single":
            self.predicted[self.mean_prob>=self.threshold] = 1
            self.predicted[self.mean_prob<self.threshold] = 0
        else:
            print("Single entropy")
            self.predicted[self.prob_rec[self.pred_entropy_single_idx][...,-1][:self.label_mask.shape[0], :self.label_mask.shape[1]]>=self.threshold] = 1
            self.predicted[self.prob_rec[self.pred_entropy_single_idx][...,-1][:self.label_mask.shape[0], :self.label_mask.shape[1]]<self.threshold] = 0

        print(np.unique(self.predicted, return_counts=True))

        self.predicted_unpad = self.predicted.copy()
        self.predicted_unpad[self.label_mask == 2] = 0
        ic(self.predicted_unpad.shape, self.predicted.shape)
        del self.predicted

    def getValidationValuesForMetrics(self):
        self.label_mask_val = self.label_mask[self.mask_tr_val == 2]
        ic(self.label_mask_val.shape)

        self.mean_prob_val = self.mean_prob[self.mask_tr_val == 2]

        self.mean_prob_val = self.mean_prob_val[self.label_mask_val != 2]
        self.label_mask_val_valid = self.label_mask_val[self.label_mask_val != 2]
        ic(self.label_mask_val_valid.shape)

        self.predicted_val = self.predicted_unpad[self.mask_tr_val == 2]
        self.predicted_val = self.predicted_val[self.label_mask_val != 2]

    def getTestValues(self):
        # test metrics
        predicted_test = self.predicted_unpad[self.mask_amazon_ts == 1]
        label_mask_current_deforestation_test = self.label_mask_current_deforestation[self.mask_amazon_ts == 1]
        label_mask_test = self.label_mask[self.mask_amazon_ts == 1]
        mean_prob_test = self.mean_prob[self.mask_amazon_ts == 1]

        ic(predicted_test.shape)

        predicted_test = utils_v1.excludeBackgroundAreasFromTest(
                predicted_test, label_mask_test)
        label_mask_current_deforestation_test = utils_v1.excludeBackgroundAreasFromTest(
                label_mask_current_deforestation_test, label_mask_test)
        mean_prob_test = utils_v1.excludeBackgroundAreasFromTest(
                mean_prob_test, label_mask_test)

        ic(predicted_test.shape)

    def removeSmallPolygons(self):
        self.removePolygons = True
        if self.removePolygons == True:
            # remove polygons smaller than 625 px
            min_polygon_area = 625 # pixels

            self.predicted_unpad, self.label_mask = _metrics.removeSmallPolygonsForMetrics(self.predicted_unpad, self.label_mask,
                min_polygon_area)
            predicted_masked, label_masked = _metrics.getTest(self.predicted_unpad, self.label_mask, self.mask_amazon_ts)

            self.predicted_test = predicted_masked
            self.label_mask_current_deforestation_test = label_masked

    def calculateMetrics(self):
        deforestationMetricsGet = True
        if deforestationMetricsGet == True:
                self.f1 = round(f1_score(self.label_mask_current_deforestation_test, self.predicted_test)*100, 2)
                self.precision = round(metrics.precision_score(self.label_mask_current_deforestation_test, self.predicted_test)*100, 2)
                self.recall = round(metrics.recall_score(self.label_mask_current_deforestation_test, self.predicted_test)*100, 2)
                if self.removePolygons == False:
                        mAP = round(metrics.average_precision_score(self.label_mask_current_deforestation_test, 
                                self.mean_prob_test)*100, 2)
                else:
                        '''
                        # Computing metrics over the test tiles
                        # mean_prob = mean_prob[:label_mask.shape[0], :label_mask.shape[1]]
                        ref1 = np.ones_like(label_mask).astype(np.float32)

                        ref1 [label_mask == 2] = 0
                        TileMask = mask_amazon_ts * ref1
                        GTTruePositives = label_mask==1

                        # Metrics for th=0.5    

                        Npoints = 50
                        Pmax = np.max(mean_prob[GTTruePositives * TileMask ==1])
                        ProbList = np.linspace(Pmax,0,Npoints)

                        metrics_ = matrics_AA_recall(ProbList, mean_prob, label_mask, mask_amazon_ts, 625)
                        print('Metrics th = 0.5: ', metrics_*100)
                        '''
                        pass
        ic(self.f1, self.precision, self.recall)
    
    def calculateMetricsValidation(self):

        f1_val = round(f1_score(self.label_mask_val_valid, self.predicted_val)*100, 2)
        precision_val = round(metrics.precision_score(self.label_mask_val_valid, self.predicted_val)*100, 2)
        recall_val = round(metrics.recall_score(self.label_mask_val_valid, self.predicted_val)*100, 2)

        mAP_val = round(metrics.average_precision_score(self.label_mask_val_valid, self.mean_prob_val)*100, 2)


        ic(f1_val, precision_val, recall_val, mAP_val)

    def calculateMAPWithoutSmallPolygons(self):

        # Computing metrics over the test tiles
        # mean_prob = mean_prob[:label_mask.shape[0], :label_mask.shape[1]]
        ref1 = np.ones_like(self.label_mask).astype(np.uint8)

        ref1 [self.label_mask == 2] = 0
        TileMask = self.mask_amazon_ts * ref1
        GTTruePositives = self.label_mask==1

        # Metrics for th=0.5    

        ProbList_05 = [0.5]

        metrics_05 = utils_v1.matrics_AA_recall(ProbList_05, self.mean_prob, self.label_mask, self.mask_amazon_ts, 625)
        print('Metrics th = 0.5: ', metrics_05*100)

    def getErrorMask(self):
        self.error_mask = np.abs(self.predicted_unpad - self.label_mask_current_deforestation)
        print(np.unique(self.error_mask, return_counts=True))
    
    def getErrorMaskToShowRGB(self):
        predicted_unpad_to_show = self.predicted_unpad.copy()

        predicted_unpad_to_show[self.label_mask == 2] = 0
        print(np.unique(predicted_unpad_to_show))
        error_mask_to_show = _metrics.getRgbErrorMask(predicted_unpad_to_show, 
                self.label_mask_current_deforestation).astype(np.uint8)
        error_mask_to_show[self.label_mask == 2] = 4 # add color for past deforestation
        self.error_mask_to_show_rgb = _metrics.saveRgbErrorMask(error_mask_to_show).astype(np.uint8)
        del error_mask_to_show
        cv2.imwrite('output/figures/Para_error_mask_to_show_rgb.png', self.error_mask_to_show_rgb)

    def setUncertainty(self):
        pass
    def getTestValues2(self):

        ic(self.label_mask.shape)
        ic(self.mask_amazon_ts.shape)
        
        self.label_mask_test = utils_v1.getTestVectorFromIm(
                self.label_mask, self.mask_amazon_ts)        
        ic(self.label_mask_test.shape)

        self.error_mask_test = utils_v1.excludeBackgroundAreasFromTest(
                utils_v1.getTestVectorFromIm(
                self.error_mask, self.mask_amazon_ts),
                self.label_mask_test) 

        
        ic(self.error_mask_test.shape)


        self.uncertainty = utils_v1.excludeBackgroundAreasFromTest(
                utils_v1.getTestVectorFromIm(
                        utils_v1.unpadIm(self.uncertainty_map, self.npad), self.mask_amazon_ts),
                self.label_mask_test)



        self.label_mask_current_deforestation_test = utils_v1.excludeBackgroundAreasFromTest(
                utils_v1.getTestVectorFromIm(
                        self.label_mask_current_deforestation, self.mask_amazon_ts),
                self.label_mask_test)

        self.predicted_test = utils_v1.excludeBackgroundAreasFromTest(
                utils_v1.getTestVectorFromIm(
                        self.predicted_unpad, self.mask_amazon_ts),
                self.label_mask_test)
                

    def getValidationValues2(self):
        self.error_mask_val = utils_v1.excludeBackgroundAreasFromTest(
                utils_v1.getTestVectorFromIm(
                self.error_mask, self.mask_tr_val, mask_return_value = 2),
                self.label_mask_val) 


        self.uncertainty_val = utils_v1.excludeBackgroundAreasFromTest(
                utils_v1.getTestVectorFromIm(
                        utils_v1.unpadIm(self.uncertainty_map, self.npad), self.mask_tr_val, mask_return_value = 2),
                self.label_mask_val)

    def getOtherUncertaintyMetrics(self):
        self.sUEO = _metrics.getSUEO(self.uncertainty,
                         self.label_mask_current_deforestation_test,
                         self.predicted_test)
        print(self.sUEO)
        self.ece_score = _metrics.ece_score( 1 - self.uncertainty, 
                                            self.predicted_test,
                                      self.label_mask_current_deforestation_test)
        print(self.ece_score)
        return self.sUEO, self.ece_score
    

    def getUncertaintyAAValues(self):



        # self.threshold_list = [0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.34, 0.36]

        # self.threshold_list = [0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.27, 0.3, 0.34, 0.36, np.max(uncertainty)-0.003]
        if self.config['uncertainty_method'] == "pred_entropy":
                self.threshold_list = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36, np.max(self.uncertainty)-0.003, np.max(self.uncertainty)-0.0015]
                self.threshold_list = [0.0025, 0.025, 0.05, 0.1, 0.2, 0.4, 
                        0.5, 0.6, 0.7, 0.8, 0.9, np.max(self.uncertainty)-0.003, np.max(self.uncertainty)-0.0015]
                
        elif self.config['uncertainty_method'] == "pred_var":
                self.threshold_list = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36]
                self.threshold_list = [x*0.13/0.36 for x in self.threshold_list] + [np.max(self.uncertainty)-0.0015, np.max(self.uncertainty)-0.0008]
        elif self.config['uncertainty_method'] == "MI":
                self.threshold_list = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36]
                self.threshold_list = [x*0.235/0.36 for x in self.threshold_list] + [np.max(self.uncertainty)-0.003, np.max(self.uncertainty)-0.0015]
        elif self.config['uncertainty_method'] == "KL":
                self.threshold_list = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36]
                self.threshold_list = [x*1.0/0.36 for x in self.threshold_list] + [np.max(self.uncertainty)-0.006, np.max(self.uncertainty)-0.003]
        elif self.config['uncertainty_method'] == "evidential":
                # self.threshold_list = [0.015, 0.03, 0.04]
                self.threshold_list = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.05, 0.08, 0.1, 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36]
                self.threshold_list = [ 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36]
                self.threshold_list = [ 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36, 0.45, 0.55, 0.65, 0.8]

                self.threshold_list = [0.13, 0.15, 0.2, 0.225, 
                        0.25, 0.27, 0.3, 0.34, 0.36, 0.45, 0.55, 0.65, 0.8]

                # self.threshold_list = [ 0.15, 0.2,  
                #          0.3, 0.35, 0.4, 0.5, 0.6, 0.7]

                # self.threshold_list = [0.015,0.1, 0.2, 0.27, 0.36, 0.45, 0.55, 0.65]

                # self.threshold_list = [x*1.0/0.36 for x in self.threshold_list] + [np.max(uncertainty)-0.006, np.max(uncertainty)-0.003]
                        
                # self.threshold_list = np.linspace(np.min(uncertainty) + 0.0015, np.max(uncertainty) - 0.0015, 19)
        print(self.threshold_list)
        self.loadThresholdMetrics = False
        if self.loadThresholdMetrics == False:
                # self.threshold_list = [0.1]
                # y_test
                ic(self.uncertainty.shape, self.label_mask_current_deforestation_test.shape)

                metrics_values = _metrics.getAA_Recall(self.uncertainty, 
                        self.label_mask_current_deforestation_test, 
                        self.predicted_test, self.threshold_list)

                # ic(metrics_values)


        self.m = {'precision_L': metrics_values[:,0],
                'recall_L': metrics_values[:,1],
                'recall_Ltotal': metrics_values[:,2],
                'AA': metrics_values[:,3],
                'precision_H': metrics_values[:,4],
                'recall_H': metrics_values[:,5],
                'UEO': metrics_values[:,6]}

        self.m['f1_L'] = 2*self.m['precision_L']*self.m['recall_L']/(self.m['precision_L']+self.m['recall_L'])
        self.m['f1_H'] = 2*self.m['precision_H']*self.m['recall_H']/(self.m['precision_H']+self.m['recall_H'])

    def getUncertaintyAAAuditedValues(self):

        if self.loadThresholdMetrics == False:
                # self.threshold_list = [0.1]
                # y_test
                ic(self.uncertainty.shape, self.label_mask_current_deforestation_test.shape)

                metric_values_audited = _metrics.getUncertaintyMetricsAudited(self.uncertainty, 
                        self.label_mask_current_deforestation_test, 
                        self.predicted_test, self.threshold_list)

                ic(metric_values_audited)

        self.m_audited = {'precision': metric_values_audited[:,0],
                'recall': metric_values_audited[:,1]}


        self.m_audited['f1'] = 2*self.m_audited['precision']*self.m_audited['recall']/(self.m_audited['precision']+self.m_audited['recall'])

    def setPlotLimsForUncertaintyAA(self):
        self.xlim = [-0.3, 12.7]
        self.xlim = [-0.1, 10.4]

        self.ylim = [0, 105]



    def plotUncertaintyAA(self):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        ax1.plot(self.m['AA']*100, self.m['precision_L']*100, 'C3-', label="Precision Low Uncertainty")
        ax1.plot(self.m['AA']*100, self.m['recall_L']*100, 'C3--', label="Recall Low Uncertainty")
        ax1.plot(self.m['AA']*100, self.m['precision_H']*100, 'C0-', label="Precision High Uncertainty")
        ax1.plot(self.m['AA']*100, self.m['recall_H']*100, 'C0--', label="Recall High Uncertainty")
        ax1.plot(self.m['AA']*100, self.m_audited['precision']*100, 'C2-', label="Precision Audited")
        ax1.plot(self.m['AA']*100, self.m_audited['recall']*100, 'C2--', label="Recall Audited")

        ax1.legend(loc="lower right")
        ax1.set_ylabel('Precision/recall (%)')
        ax1.set_xlabel('Audit Area (%)')
        ax1.set_ylim(self.ylim)
        ax1.set_xlim(self.xlim)
        ax1.grid()

        xs = [0, 120]
        ax1.vlines(x = 3, ymin = 0, ymax = max(xs),
                colors = (0.2, 0.2, 0.2),
                label = 'vline_multiple - full height')

        ax2.plot(range(int(self.xlim[0]), int(self.xlim[1] + 2)), 
            np.ones(int(self.xlim[1] + 2)) * self.f1, 
            'C1:', label="F1 No Uncertainty")

        ax2.plot(self.m['AA']*100, self.m['f1_L']*100, 'C3-', label="F1 Low Uncertainty")
        ax2.plot(self.m['AA']*100, self.m['f1_H']*100, 'C0-', label="F1 High Uncertainty")
        ax2.plot(self.m['AA']*100, self.m_audited['f1']*100, 'C2-', label="F1 Audited")

        ax2.legend(loc="lower right")
        ax2.set_ylabel('F1 score (%)')
        ax2.set_xlabel('Audit Area (%)')
        ax2.set_ylim(self.ylim)
        ax2.set_xlim(self.xlim)

        ax2.grid()

        xs = [0, 120]
        ax2.vlines(x = 3, ymin = 0, ymax = max(xs),
                colors = (0.2, 0.2, 0.2),
                label = '3% AA')

        ax3.plot(np.asarray(self.threshold_list), self.m['AA']*100, label="AA")
        ax3.set_ylabel('Audit Area (%)')
        ax3.set_xlabel('Uncertainty Threshold')
        ax3.grid()
        ax3.set_ylim(self.xlim)

        self.xlim_adjusted = ax3.get_xlim()
        ax3.hlines(y = 3, xmin = self.xlim_adjusted[0], xmax = self.xlim_adjusted[1],
                colors = (0.2, 0.2, 0.2),
                label = '3% AA')

        ax3.set_xlim(self.xlim_adjusted)

        # if save_figures == True:
        if True:
            plt.savefig('output/figures/recall_precision_f1_AA.png', dpi=150, bbox_inches='tight')

    def plotUEO(self): 

        plt.plot(self.m['AA']*100, self.m['UEO'], label="UEO") 
        plt.grid() 
        plt.xlabel('Audit Area (%)') 
        plt.ylabel('UEO (%)') 
        plt.xlim(self.xlim)
        plt.ylim([0, 0.4])
        plt.savefig('output/figures/ueo.png', dpi=150, bbox_inches='tight')

    def getOptimalUncertaintyThreshold(self, AA = 0.03, bound = 0.0015):

        def getAAFromUncertaintyThreshold(threshold): 
            print(threshold)
            metrics_values2 = _metrics.getAA_Recall(self.uncertainty, 
                            self.label_mask_current_deforestation_test, 
                            self.predicted_test, [threshold])
            return np.abs(AA - metrics_values2[:,3].squeeze())

        bounds = (bound, np.max(self.uncertainty)-0.0015)
        ic(bounds)
        minimum = optimize.minimize_scalar(getAAFromUncertaintyThreshold, 
            method='bounded', bounds=bounds, tol=0.0001)
        self.threshold_optimal = minimum.x
        ic(self.threshold_optimal)
    
    def getUncertaintyMetricsFromOptimalThreshold(self, get_f1=True):

        self.metric_values_optimal = _metrics.getAA_Recall(self.uncertainty, 
                                self.label_mask_current_deforestation_test, 
                                self.predicted_test, [self.threshold_optimal])

        self.metric_values_audited_optimal = _metrics.getUncertaintyMetricsAudited(self.uncertainty, 
                self.label_mask_current_deforestation_test, 
                self.predicted_test, [self.threshold_optimal])

        self.m_optimal = {'precision_L': self.metric_values_optimal[:,0],
                'recall_L': self.metric_values_optimal[:,1],
                'recall_Ltotal': self.metric_values_optimal[:,2],
                'AA': self.metric_values_optimal[:,3],
                'precision_H': self.metric_values_optimal[:,4],
                'recall_H': self.metric_values_optimal[:,5],
                'UEO': self.metric_values_optimal[:,6]}

        self.m_audited_optimal = {'precision': self.metric_values_audited_optimal[:,0],
                'recall': self.metric_values_audited_optimal[:,1]}

        self.m_optimal['f1_L'] = 2*self.m_optimal['precision_L']*self.m_optimal['recall_L']/(self.m_optimal['precision_L']+self.m_optimal['recall_L'])
        self.m_optimal['f1_H'] = 2*self.m_optimal['precision_H']*self.m_optimal['recall_H']/(self.m_optimal['precision_H']+self.m_optimal['recall_H'])
        if get_f1 == True:
            self.m_optimal['f1'] = self.f1
        self.m_audited_optimal['f1'] = 2*self.m_audited_optimal['precision']*self.m_audited_optimal['recall']/(self.m_audited_optimal['precision']+self.m_audited_optimal['recall'])

        ic(self.m_optimal)
        ic(self.m_audited_optimal)

        # np.save('m_optimal_{}.npy'.format(self.grid_idx), self.m_optimal)
        # np.save('m_audited_optimal_{}.npy'.format(self.grid_idx), self.m_audited_optimal)
        return {'metrics': self.m_optimal, 'metrics_audited': self.m_audited_optimal, 'exp': self.exp}
        
    def saveResults(self):

        results = {
            "metrics": {
                "f1": self.f1,
                "precision": self.precision,
                "recall": self.recall
            },
            "m_optimal": self.m_optimal,
            "m_audited_optimal": self.m_audited_optimal,
        }
        np.save('results_{}.npy'.format(self.grid_idx), results)


    def plotLossTerms(self):
        pass
    def plotAnnealingCoef(self):
        pass

    def run_predictor_repetition(self, uncertainty_methods=['pred_entropy', 'pred_var', 'MI', 'KL']):
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
        
        # uncertainty_methods = 
        results = {}
        for uncertainty_method in uncertainty_methods:
            self.config['uncertainty_method'] = uncertainty_method
            self.setUncertainty()
            self.getValidationValues2()
            self.getTestValues2()
            self.getOptimalUncertaintyThreshold()
            results[uncertainty_method] = self.getUncertaintyMetricsFromOptimalThreshold()
        '''
        min_metric = np.inf
        max_metric = 0
        for idx in range(self.config['inference_times']):
            self.pred_entropy_single_idx = idx
            self.config['uncertainty_method'] = "pred_entropy_single"
            self.setUncertainty()
            self.getValidationValues2()
            self.getTestValues2()
            self.getOptimalUncertaintyThreshold()
            results_tmp = self.getUncertaintyMetricsFromOptimalThreshold()
            metric = results_tmp['metrics']['f1']
            if metric > max_metric:
                max_metric = metric
                results["pred_entropy_single_max"] = results_tmp
            if metric < min_metric:
                min_metric = metric
                results["pred_entropy_single_min"] = results_tmp
        
        '''
        print("results",  results)
        return results


