from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn import metrics
class Monitor(Callback):
    def __init__(self, validation, patience, classes, sample_validation_store=False):   
        super(Monitor, self).__init__()
        self.validation = validation 
        self.patience = patience
        self.best_weights = None
        self.classes = classes
        self.f1_history = []
        self.oa_history = []
    def on_train_begin(self, logs={}):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0
    def on_epoch_begin(self, epoch, logs={}):        
        self.pred = []
        self.targ = []

    def on_epoch_end(self, epoch, logs={}):

        self.getValidationData()
        f1 = np.round(metrics.f1_score(self.targ, self.pred, average=None)*100,2)
        precision = np.round(metrics.precision_score(self.targ, self.pred, average=None)*100,2)
        recall= np.round(metrics.recall_score(self.targ, self.pred, average=None)*100,2)

        #update the logs dictionary:
        mean_f1 = np.sum(f1)/self.classes
        logs["mean_f1"]=mean_f1

        self.f1_history.append(mean_f1)
        
        print(f' — val_f1: {f1}\n — val_precision: {precision}\n — val_recall: {recall}')
        print(f' — mean_f1: {mean_f1}')

    def getValidationData(self):

        for batch_index in range(len(self.validation)):
            val_targ = self.validation[batch_index][1]   
            val_pred = self.model.predict(self.validation[batch_index][0])
            print("val_pred.shape", val_pred.shape) # was programmed to get two outputs> classif. and depth
            print("val_targ.shape", val_targ.shape) # was programmed to get two outputs> classif. and depth
            print("len(self.validation[batch_index][1])",
                len(self.validation[batch_index][1])) # was programmed to get two outputs> classif. and depth

            val_prob = val_pred.copy()
            val_predict = np.argmax(val_prob,axis=-1)
        
            val_targ = np.squeeze(val_targ)
            self.pred.extend(val_predict)
            self.targ.extend(val_targ)       