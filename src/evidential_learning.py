import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pdb
print(f"Tensorflow ver. {tf.__version__}")

class EvidentialLearning:
    def __init__(self):
        self.an_ = 0
        self.im_len = 128
        self.num_classes = 3
        
    def KL(self, alpha):
        beta=tf.constant(np.ones((self.im_len,self.im_len,self.num_classes)),dtype=tf.float32,shape=(self.im_len,self.im_len,self.num_classes))
        
        S_alpha = tf.reduce_sum(alpha,axis=3,keepdims=True)
        S_beta = tf.reduce_sum(beta,axis=2,keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=3,keepdims=True)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=2,keepdims=True) - tf.math.lgamma(S_beta)
        
        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)
        
        kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=3,keepdims=True) + lnB + lnB_uni
    
        return kl

    def exp_evidence(self, logits): 
        return tf.exp(tf.clip_by_value(logits/10,-10,10))

    def DL(self, y_truth,y_pred):
        S = tf.reduce_sum(y_pred, axis=3,keepdims=True)
        p = tf.math.divide_no_nan(y_pred,S)
        E = y_pred - 1

        intersection = tf.reduce_sum(tf.multiply(y_truth,p))
        dice = tf.math.divide_no_nan((2*intersection + 0.00001),(tf.reduce_sum(y_truth) + tf.reduce_sum(p) + 0.00001))
        A = 1 - dice

        alp = tf.add(tf.multiply(E,tf.subtract(1.,y_truth)),1) 
        B = self.KL(alp)
    
        loss = A + B
        return loss

    def categorical_crossentropy_envidential_learning(self, y_truth,y_pred):
        S = tf.reduce_sum(y_pred, axis=3,keepdims=True)
        dgS = tf.math.digamma(S)
        dgalpha = tf.math.digamma(y_pred)

        A = tf.reduce_sum(tf.multiply(y_truth,tf.subtract(dgS,dgalpha)),axis=3,keepdims=True)
        E = y_pred - 1
        alp = tf.add(tf.multiply(E,tf.subtract(1.,y_truth)),1) 
        B = self.KL(alp)
        loss = tf.reduce_mean(A) + (self.an_*B)             
        
        return loss

    def mse_loss(self, y_truth,y_pred):
        S = tf.reduce_sum(y_pred, axis=3,keepdims=True)
        E = y_pred - 1
        prob = tf.math.divide_no_nan(y_pred,S)

        A = tf.reduce_sum((y_truth-prob)**2, axis=3, keepdims=True) 
        B = tf.reduce_sum(y_pred*(S-y_pred)/(S*S*(S+1)), axis=3, keepdims=True) 
        
        #annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
        
        alp = tf.add(tf.multiply(E,tf.subtract(1.,y_truth)),1) 
        C =  self.KL(alp)
        return (A + B) + (0.3*C)

    def evidential_accuracy(self, y_truth,y_pred):
        pred = tf.argmax(y_pred,axis=3)
        truth = tf.argmax(y_truth,axis=3)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32),(-1,1))
        acc = tf.reduce_mean(match)
        return acc   
    
    def updateAnnealingCoeficient(self, epoch):
        self.an_ = np.minimum([1.0],[(float(epoch)/75.0)])
    
class DirichletLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs, **kwargs):
    super(DirichletLayer, self).__init__(**kwargs)
    self.num_outputs = num_outputs

  def get_config(self):
        config = {
            "num_outputs" : self.num_outputs
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
  
  def call(self, inputs):
    evidence = tf.exp(tf.clip_by_value(inputs,-10,10)) #Remover a divisao por 10
    alpha = evidence + 1
    #S = tf.reduce_sum(alpha, axis=3, keepdims=True)
    #m = tf.math.divide_no_nan(evidence,S) # m = bk
    
    return alpha

def alpha_to_probability_and_uncertainty(alpha):
    print("alpha.shape", alpha.shape)

    S = np.sum(alpha, axis = -1)
    print("S.shape", S.shape)
    K = np.shape(alpha)[-1]
    # print("K.shape", K.shape)
    print("K", K)
    u = K / S
    print("u.shape", u.shape)
    belief = (alpha - 1) / np.expand_dims(S, axis = -1)
    print("belief.shape", belief.shape)
    return belief, u

