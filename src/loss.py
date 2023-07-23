# Import libraries
import os
import sys
import time
import math
import random
import numpy as np
from PIL import Image
import tensorflow as tf
#from libtiff import TIFF
import skimage.morphology 
from osgeo import ogr, gdal
from scipy import ndimage
#import tifffile
import matplotlib.pyplot as plt
from skimage.filters import rank
from sklearn.utils import shuffle
from skimage.morphology import disk
from skimage.transform import resize
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix
from skimage.util.shape import view_as_windows
from sklearn.metrics import average_precision_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def weighted_categorical_crossentropy(weights):
		"""
		A weighted version of keras.objectives.categorical_crossentropy
		
		Variables:
			weights: numpy array of shape (C,) where C is the number of classes
		
		Usage:
			weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
			loss = weighted_categorical_crossentropy(weights)
			model.compile(loss=loss,optimizer='adam')
		"""
		
		weights = K.variable(weights)
			
		def loss(y_true, y_pred):
			# scale predictions so that the class probas of each sample sum to 1
			y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
			# clip to prevent NaN's and Inf's
			y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
			loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)

			print("K.int_shape(y_pred)", K.int_shape(y_pred))
			print("K.int_shape(y_true)", K.int_shape(y_true))
			print("K.int_shape(loss)", K.int_shape(loss))
			print("K.int_shape(weights)", K.int_shape(weights))

			loss = loss * weights 
			print("K.int_shape(loss)", K.int_shape(loss))
			loss = - K.mean(loss, -1)
			print("K.int_shape(loss)", K.int_shape(loss))

			return loss
		return loss


def weighted_categorical_crossentropy(weights):
	"""
	A weighted version of keras.objectives.categorical_crossentropy
	
	Variables:
		weights: numpy array of shape (C,) where C is the number of classes
	
	Usage:
		weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
		loss = weighted_categorical_crossentropy(weights)
		model.compile(loss=loss,optimizer='adam')
	"""
	
	weights = K.variable(weights)
		
	def loss(y_true, y_pred):
		# y_true is of shape (N, H, W, C) with C the number of classes = 3. Ignore last index (ignore_index=2)
		# y_pred is of shape (N, H, W, C) with C the number of classes = 2			
		# scale predictions so that the class probas of each sample sum to 1
		# y_true = y_true[:,:,:,0:2]
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)

		print("K.int_shape(y_pred)", K.int_shape(y_pred))
		print("K.int_shape(y_true)", K.int_shape(y_true))
		print("K.int_shape(loss)", K.int_shape(loss))
		print("K.int_shape(weights)", K.int_shape(weights))

		loss = loss * weights 
		print("K.int_shape(loss)", K.int_shape(loss))
		loss = - K.mean(loss, -1)
		print("K.int_shape(loss)", K.int_shape(loss))

		return loss
	return loss



def weighted_cross_entropy_loss(loss_weights):
# def weighted_categorical_crossentropy(loss_weights):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    loss_weights = K.variable(loss_weights)

    def loss_fn(y_true, y_pred):
        # Adjust the number of classes in y_pred
        adjusted_num_classes = 2
        adjusted_y_true = y_true[:, :, :, :-1]
        
        # Apply loss weights
        per_pixel_loss = loss_object(adjusted_y_true, y_pred) * loss_weights

        # Ignore specific indices
        mask = tf.cast(tf.not_equal(tf.argmax(adjusted_y_true, axis=-1), adjusted_num_classes), dtype=tf.float32)
        per_pixel_loss *= mask

        # Calculate mean loss
        loss = tf.reduce_mean(per_pixel_loss)
        return loss

    return loss_fn