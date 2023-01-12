
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

class TrainerEvidential(Trainer):
    def __init__(self, config, dataset, patchesHandler, grid_idx=0):
        super().__init__(config, dataset, patchesHandler, grid_idx=grid_idx)
        self.annealing_step  = config['Uncertainty']['annealing_step']
        self.times = 1
        self.network_architecture = utils_v1.build_resunet
        # 10*117
        # self.annealing_step  = 10*375
        # self.annealing_step  = 10*375/2

        # self.annealing_step  = 10*375/4
    def train(self):


        # evidential
        class_n = 3

        def KL(alpha, K):
            beta=tf.constant(np.ones((1,K)),dtype=tf.float32)
            S_alpha = tf.reduce_sum(alpha,axis=-1,keepdims=True)
            
            KL = tf.reduce_sum((alpha - beta)*(tf.math.digamma(alpha)-tf.math.digamma(S_alpha)),axis=-1,keepdims=True) + \
                tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=-1,keepdims=True) + \
                tf.reduce_sum(tf.math.lgamma(beta),axis=-1,keepdims=True) - tf.math.lgamma(tf.reduce_sum(beta,axis=-1,keepdims=True))
            return KL

        # KL_reg_monitor = K.variable(0.0)

        def loss_eq5(p, alpha, K, global_step, annealing_step):
            S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
            loglikelihood = tf.reduce_sum((p-(alpha/S))**2, axis=-1, keepdims=True) + tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=-1, keepdims=True)
            #global_step = tf.compat.v1.train.get_global_step
            KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
            # tf.keras.backend.set_value(KL_reg_monitor, tf.keras.backend.get_value(KL_reg))
            return loglikelihood + KL_reg
        '''
        class EpochTracker(Callback):
            def __init__(self):
                super(EpochTracker, self).__init__()
                self.epoch_n = 0 

            def on_epoch_begin(self, epoch, logs={}):
                #if epoch%20 == 0:   
                K.set_value(self.epoch_n, epoch)
                print("Setting alpha to =", str(epoch))
        epochTracker = EpochTracker()
        '''

        import tensorflow.keras.backend as K


        global_step = K.variable(0.0)

        # def loss_evidential(weights):
        def loss_evidential():

            # init the tensor with current epoch, to be updated during training, and define var in scope
            # self.global_step = K.variable(0.0)
            # global_step = self.global_step  
            def loss(y_true, y_pred):  
                evidence = tf.nn.relu(y_pred)

                alpha = evidence + 1
                u = class_n / tf.reduce_sum(alpha, axis= -1, keepdims=True)

                print("alpha", alpha)
                print("u", u)
                prob = alpha / tf.reduce_sum(alpha, axis = -1, keepdims=True) 

                Y = y_true
                # loss = loss_eq5(Y, alpha, class_n, global_step, 30) # 10*34
                # loss = loss_eq5(Y, alpha, class_n, global_step, 40) # 10*34
                loss = loss_eq5(Y, alpha, class_n, global_step, self.annealing_step) # 10*3753/32

                #    loss = loss_eq5(Y, alpha, class_n, global_step, 15) # 10*34
                #    loss = loss_eq5(Y, alpha, class_n, global_step, 5) # 10*34
                #    loss = loss_eq5(Y, alpha, class_n, global_step, 60) # 10*34
                # loss = loss * weights
                loss = tf.reduce_mean(loss)
                return loss
            return loss
        class GetCurrentEpoch(Callback):
            """get the current epoch to pass it within the loss function.
            # Arguments
                global_step: The tensor withholding the current epoch on_epoch_begin
            """

            def __init__(self, global_step):
                super().__init__()
                self.global_step = global_step
                # self.KL_reg_monitor = KL_reg_monitor
                self.epoch = 0
            def on_batch_begin(self, step, logs=None):
                new_step = step + self.epoch * step
                # Set new value
                K.set_value(self.global_step, new_step)

            def on_epoch_begin(self, epoch, logs=None):
                # K.set_value(self.epoch, epoch)
                self.epoch = epoch
                print("self.global_step", K.get_value(self.global_step))
                # print("KL_reg_monitor", K.get_value(self.KL_reg_monitor))

        import tensorflow.keras.backend as K

        def loss_eq5_metric(p, alpha, K, global_step, annealing_step):
            S = tf.reduce_sum(alpha, axis=-1, keepdims=True)
            loglikelihood = tf.reduce_sum((p-(alpha/S))**2, axis=-1, keepdims=True) + tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=-1, keepdims=True)
            #global_step = tf.compat.v1.train.get_global_step
            KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
            # tf.keras.backend.set_value(KL_reg_monitor, tf.keras.backend.get_value(KL_reg))
            print("K.int_shape(KL_reg)", KL_reg)
            return loglikelihood + KL_reg, loglikelihood, KL_reg

        def loss_eq4_metric(p, alpha, K, global_step, annealing_step):
            loglikelihood = tf.reduce_mean(tf.reduce_sum(p * (tf.math.digamma(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.math.digamma(alpha)), 1, keepdims=True))
            KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
            return loglikelihood + KL_reg, loglikelihood, KL_reg

        def loss_eq3_metric(p, alpha, K, global_step, annealing_step):
            loglikelihood = tf.reduce_mean(tf.reduce_sum(p * (tf.math.log(tf.reduce_sum(alpha, axis=1, keepdims=True)) - tf.math.log(alpha)), 1, keepdims=True))
            KL_reg =  tf.minimum(1.0, tf.cast(global_step/annealing_step, tf.float32)) * KL((alpha - 1)*(1-p) + 1 , K)
            return loglikelihood + KL_reg, loglikelihood, KL_reg

        def evidence_get(y_pred):
            evidence = tf.nn.relu(y_pred)

            alpha = evidence + 1
            u = class_n / tf.reduce_sum(alpha, axis= -1, keepdims=True)

            print("alpha", alpha)
            print("u", u)
            prob = alpha / tf.reduce_sum(alpha, axis = -1, keepdims=True) 

            return alpha, u

        loss_metric = loss_eq5_metric
        def KL_term(y_true, y_pred):
            alpha, u = evidence_get(y_pred)
            Y = y_true

            _, _, KL_reg = loss_metric(Y, alpha, class_n, global_step, self.annealing_step)

            KL_reg = tf.reduce_mean(KL_reg)
            return KL_reg

        def loglikelihood_term(y_true, y_pred):
            alpha, u = evidence_get(y_pred)
            Y = y_true

            _, loglikelihood, _ = loss_metric(Y, alpha, class_n, global_step, self.annealing_step)

            loglikelihood = tf.reduce_mean(loglikelihood)
            return loglikelihood

        def acc(y_true, y_pred):
            logits = y_pred
            Y = y_true
            evidence = tf.nn.relu(y_pred)
            match = tf.reshape(tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(Y, -1)), tf.float32),(-1,1))
            acc = tf.reduce_mean(match)
            return acc

        def evidential_success(y_true, y_pred):
            logits = y_pred
            Y = y_true
            evidence = tf.nn.relu(y_pred)
            match = tf.reshape(tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(Y, -1)), tf.float32),(-1,1))
            mean_ev_succ = tf.reduce_sum(tf.reshape(tf.reduce_sum(evidence,-1, keepdims=True), (-1,1)) * match) / tf.reduce_sum(match+1e-20)
            return mean_ev_succ
        def evidential_fail(y_true, y_pred):
            logits = y_pred
            Y = y_true
            evidence = tf.nn.relu(y_pred)
            match = tf.reshape(tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(Y, -1)), tf.float32),(-1,1))
            mean_ev_fail = tf.reduce_sum(tf.reshape(tf.reduce_sum(evidence,-1, keepdims=True), (-1,1)) * (1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
            return mean_ev_fail

        def annealing_coef(y_true, y_pred):
            return tf.minimum(1.0, tf.cast(global_step/self.annealing_step, tf.float32))

        def global_step_get(y_true, y_pred):
            return tf.cast(global_step, tf.float32)

        def annealing_step_get(y_true, y_pred):
            return tf.cast(self.annealing_step, tf.float32)
            

        metrics_all = []

        for tm in range(0,self.times):
            print('time: ', tm)

            rows = self.patch_size
            cols = self.patch_size
            adam = Adam(lr = 1e-3 , beta_1=0.9)
            
    #         loss = loss.weighted_categorical_crossentropy(weights)
            loss = loss_evidential()

            input_shape = (rows, cols, self.channels)
            self.model = self.network_architecture(input_shape, self.nb_filters, self.class_n, last_activation=None)
            
            self.model.compile(optimizer=adam, loss=loss, metrics=['accuracy', KL_term, loglikelihood_term, 
                evidential_success, evidential_fail, acc, annealing_coef, global_step_get, annealing_step_get])
            self.model.summary()

            earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
            checkpoint = ModelCheckpoint(self.path_models+ '/' + self.method +'_'+str(tm)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
            get_global_step = GetCurrentEpoch(global_step=global_step)
            
            callbacks_list = [earlystop, checkpoint, get_global_step]
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

    def plotLossTerms(self):
        self.logger.plotLossTerms(self.history)

    def plotAnnealingCoef(self):
        self.logger.plotAnnealingCoef(self.history)
    def getMeanProb(self):
        self.mean_prob = self.prob_rec
    def preprocessProbRec(self):
        self.prob_rec = np.expand_dims(self.prob_rec, axis = -1)


    def infer(self):
        self.h, self.w, self.c = self.image1_pad.shape
        self.c = self.channels
        self.patch_size_rows = self.h//self.n_rows
        self.patch_size_cols = self.w//self.n_cols
        self.num_patches_x = int(self.h/self.patch_size_rows)
        num_patches_y = int(self.w/self.patch_size_cols)

        ic(self.path_models+ '/' + self.method +'_'+str(0)+'.h5')
        model = utils_v1.load_model(self.path_models+ '/' + self.method +'_'+str(0)+'.h5', compile=False)
        self.class_n = 3

        if self.config["loadInference"] == False:
            if self.config["save_probabilities"] == False:
                # self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.class_n, inference_times), dtype = np.float32)
                self.prob_rec = np.zeros((self.image1_pad.shape[0],self.image1_pad.shape[1], self.class_n), dtype = np.float32)

        #    new_model = utils_v1.build_resunet(input_shape=(self.patch_size_rows,self.patch_size_cols, c), 
        #        nb_filters = nb_filters, n_classes = self.class_n, dropout_seed = None)
            new_model = self.network_architecture(input_shape=(self.patch_size_rows,self.patch_size_cols, self.c), 
                nb_filters = self.nb_filters, n_classes = self.class_n, last_activation=None)

            for l in range(1, len(model.layers)):
                new_model.layers[l].set_weights(model.layers[l].get_weights())
            
            self.patchesHandler.class_n = self.class_n

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
                        'dropout_seed': self.inference_times}
                    '''
                    prob_reconstructed, self.u_reconstructed = self.patchesHandler.infer(
                            new_model, self.image1_pad, self.h, self.w, 
                            # model, self.image1_pad, h, w, 
                            self.num_patches_x, num_patches_y, self.patch_size_rows, 
                            self.patch_size_cols)
                            # patch_size_cols, a = args_network)
                            
                    ts_time =  time.time() - start_test

                    if self.config["save_probabilities"] == True:
                        np.save(self.path_maps+'/'+'prob_'+str(tm)+'.npy',prob_reconstructed) 
                    else:
                        self.prob_rec = prob_reconstructed.copy()
                    
                    metrics_all.append(ts_time)
                    del prob_reconstructed
                metrics_ = np.asarray(metrics_all)
                # Saving test time
                np.save(self.path_exp+'/metrics_ts.npy', metrics_)
        del self.image1_pad


    # to-do: pass to predictor. to do that, pass data to dataset class (dataset.image_stack, dataset.label, etc)


    def setUncertainty(self):
        self.uncertainty_map = self.u_reconstructed