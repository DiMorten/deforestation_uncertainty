import sys
import numpy as np
from icecream import ic
import pdb
import scipy
sys.path.append("..")
from src.paths import PathsPara, PathsMT
import utils_v1
import skimage
class Dataset():
	def getTrainValTestMasks(self, mask_tiles):
		tiles_tr, tiles_val, tiles_ts = self.calculateTiles()

		print('Training tiles: ', tiles_tr)
		print('Validation tiles: ', tiles_val)
		print('Test tiles: ', tiles_ts)

		# Training and validation mask
		mask_tr_val = np.zeros((mask_tiles.shape)).astype('uint8')

		for tr_ in tiles_tr:
			mask_tr_val[mask_tiles == tr_] = 1

		for val_ in tiles_val:
			mask_tr_val[mask_tiles == val_] = 2

		mask_amazon_ts = np.zeros((mask_tiles.shape)).astype('uint8')
		for ts_ in tiles_ts:
			mask_amazon_ts[mask_tiles == ts_] = 1
		return mask_tr_val, mask_amazon_ts        
	def getLabelCurrentDeforestation(self, label_mask, selected_class = 1):
		label_mask_current_deforestation = label_mask.copy()
		label_mask_current_deforestation[label_mask_current_deforestation == selected_class] = 10
		label_mask_current_deforestation[label_mask_current_deforestation != 10] = 0
		label_mask_current_deforestation[label_mask_current_deforestation == 10] = 1
		return label_mask_current_deforestation.astype(np.uint8)
	def createDistMap(self, past_ref, th_sup = 800):
		print('Generating distance map ... ')
		# import scipy

		dist_past = past_ref.copy()
		dist_matrix = scipy.ndimage.distance_transform_edt(dist_past == 0)
		dist_matrix[dist_matrix >= th_sup] = th_sup
		dist_norm = (dist_matrix-np.min(dist_matrix))/(np.max(dist_matrix)-np.min(dist_matrix))
		return dist_norm     
	def getSeparateLabelsForClasses1And2(self, label):

		label_class1 = label.copy()
		label_class1[label_class1 == 2] = 0

		label_class2 = label.copy()
		label_class2[label_class2 == 1] = 0
		label_class2[label_class2 == 2] = 1
		return label_class1, label_class2
	def removeBorderBufferFromLabel(self, label, borderBuffer):
		# Creation of border buffer for pixels not considered

		label_class1, label_class2 = self.getSeparateLabelsForClasses1And2(label)
		image_ref_ = label_class1.copy()
		im_dilate = skimage.morphology.dilation(image_ref_, skimage.morphology.disk(borderBuffer))
		im_erosion = skimage.morphology.erosion(image_ref_, skimage.morphology.disk(borderBuffer))
		inner_buffer = image_ref_ - im_erosion
		inner_buffer[inner_buffer == 1] = 2
		outer_buffer = im_dilate-image_ref_
		outer_buffer[outer_buffer == 1] = 2
		
		# 1 deforestation, 2 unknown
		image_ref_[outer_buffer + inner_buffer == 2 ] = 2

		image_ref_[label_class2 == 1] = 2
		return image_ref_   
class Para(Dataset):
	def __init__(self):
		self.paths = PathsPara()

		self.grid_x, self.grid_y = 5,4

		self.label_filename = 'mask_label_17730x9203.npy'
	def loadLabel(self):
		label = np.load(self.paths.label + self.label_filename).astype('uint8')
		return label
	def loadInputImage(self):
		return np.load(self.paths.optical_im + 'optical_im.npy').astype('float32')
	def calculateTiles(self):
		# Defining tiles for training, validation and test sets
		tiles_tr = [1,3,5,8,11,13,14,20] 
		tiles_val = [6,19]
		tiles_ts = list(set(np.arange(self.grid_x * self.grid_y)+1)-set(tiles_tr)-set(tiles_val))
		return tiles_tr, tiles_val, tiles_ts
	def loadPastDeforestationLabel(self):
		label_past_deforestation = self.loadLabel()

		label_past_deforestation[label_past_deforestation == 1] = 0
		label_past_deforestation[label_past_deforestation == 2] = 1

		return label_past_deforestation

class ParaDistanceMap(Para):
	def loadInputImage(self):
		image_stack = super().loadInputImage()
		image_stack = self.addNpyBandToInput(image_stack, 
				self.paths.distance_map_past_deforestation)
		image_stack = self.addNpyBandToInput(image_stack, 
				self.paths.distance_map_past_deforestation_2018)
		image_stack = self.addNpyBandToInput(image_stack, 
				self.paths.distance_map_past_deforestation_2017)
		image_stack = self.addNpyBandToInput(image_stack, 
				self.paths.distance_map_past_deforestation_2016)

		ic(image_stack.shape)
		return image_stack  
	def addNpyBandToInput(self, image_stack, path):
		band = np.load(path).astype(np.float32)
		band = np.expand_dims(band, axis = -1)
		image_stack = np.concatenate((band, image_stack), axis = -1)  
		return image_stack    

class ParaDeforestationTime(Para):
	def __init__(self, addPastDeforestationInput = True):
		self.addPastDeforestationInput = addPastDeforestationInput
		super().__init__()

	def loadInputImage(self):
		image_stack = super().loadInputImage()
		if self.addPastDeforestationInput == True:
			image_stack = self.addDeforestationTime(image_stack)
		# image_stack = self.addPastDeforestation(image_stack)
		ic(image_stack.shape)
		return image_stack  
	def addDeforestationTime(self, image_stack):
		deforestation_time = np.load(self.paths.label + 'deforestation_time_normalized_2018_2019.npy') # has past deforestation up to 2018
		# 2018 = 0, 2017 = 1, 2016 = 2,....
		ic(deforestation_time.dtype, deforestation_time.shape)
		self.usePastDeforestationWithoutDistance = False
		if self.usePastDeforestationWithoutDistance == True:
			ic(np.unique(deforestation_time, return_counts=True))
			deforestation_time[deforestation_time>0] = 1
		ic(np.unique(deforestation_time, return_counts=True))

		
		ic(deforestation_time.shape, image_stack.shape)
		image_stack = np.concatenate((deforestation_time, image_stack), axis = -1)
		del deforestation_time  
		return image_stack

class ParaMultipleDates(ParaDeforestationTime):
	def __init__(self, dates = [2017, 2018, 2019], addPastDeforestationInput = True, borderBuffer = 0):
		super().__init__(addPastDeforestationInput)
		self.dates = dates
		# self.date_ids = [0,1]
		self.date_ids = range(len(dates[:-1]))
		ic(list(self.date_ids))
		self.image_channels = []

		if self.addPastDeforestationInput == True:
			# self.image_channels = [[0,] + list(range(2,22)),
			# 	[1,] + list(range(12,32))]
			for date_id in self.date_ids:
				self.image_channels.append([date_id,] + list(range(date_id * 10 + len(self.date_ids), 
					date_id * 10 + 20 + len(self.date_ids)))) 
			# image_channels_check = [[0,] + list(range(2,22)),
			# 	[1,] + list(range(12,32))]
			# assert image_channels_check == self.image_channels
			
		else:
			for date_id in self.date_ids:
				self.image_channels.append([date_id,] + list(range(date_id * 10, 
					date_id * 10 + 20))) 
		ic(self.image_channels)

		self.borderBuffer = borderBuffer
		# ic(self.image_channels == image_channels_check)
	def loadPastDeforestationBefore2008(self):
		label_past_deforestation_before_2008 = utils_v1.load_tiff_image(
			self.paths.deforestation_before_2008)
		return label_past_deforestation_before_2008
	def loadInputImage(self):
		# self.addPastDeforestationInput = False
		# image_stack = super().loadInputImage()
		# self.addPastDeforestationInput = True
		image_stack = []
		for date in self.dates:
			image_stack.append(
				np.load(self.paths.optical_im_past_dates[date] + 'optical_im.npy').astype('float32')
			)
		image_stack = np.concatenate(image_stack, axis = -1)
		'''
		image_stack = np.concatenate((
			np.load(self.paths.optical_im_past_dates[2017] + 'optical_im.npy').astype('float32'),
			image_stack),
			axis = -1)
		'''

		if self.addPastDeforestationInput == True:
			image_stack = self.addDeforestationTime(image_stack)        
		ic(image_stack.shape)
		return image_stack

	def addDeforestationTime(self, image_stack):
		'''
		image_stack = super().addDeforestationTime(image_stack)
		print("2")
		ic(image_stack.shape)
		'''
		deforestation_times = []
		for date in self.dates[:-1]:
			deforestation_times.append(
				np.load(self.paths.deforestation_time[date]).astype(np.float32)
			)
		deforestation_times = np.concatenate(deforestation_times, axis = -1)
		image_stack = np.concatenate((deforestation_times, 
			image_stack), axis = -1)
		del deforestation_times
		
		return image_stack

	def loadLabel(self):
		
		label_per_date = []
		for date in self.dates[1:]:
			label = self.loadLabelFromDate(date)
			label = self.addCloudMaskToLabel(label, date)
			if self.borderBuffer > 0:
				label = self.removeBorderBufferFromLabel(label, self.borderBuffer)
			if date == 2019:
				label = self.loadLabelFromProject()

			label_per_date.append(
				np.expand_dims(label, axis = -1)
			)
		label_per_date = np.concatenate(label_per_date, axis = -1)

		'''
		label_past_date = self.loadLabelFromDate(2018)
		label_current_date = super().loadLabel()

		ic(np.unique(label_current_date, return_counts=True),
			np.unique(label_past_date, return_counts=True))

		label_per_date = np.concatenate((
			np.expand_dims(label_past_date, axis = -1),
			np.expand_dims(label_current_date, axis = -1)),
			axis = -1)
		del label_past_date, label_current_date
		'''
		ic(label_per_date.shape)
		return label_per_date

	def loadLabelFromDate(self, date):
		deforestation_past_years = utils_v1.load_tiff_image(
			self.paths.deforestation_past_years)
		label = np.zeros_like(deforestation_past_years, dtype = np.uint8)
		print("Loaded deforestation past years")
		ic(np.unique(deforestation_past_years, return_counts=True))
		print("Label where deforestation past years is actual date (2018) = 1")
		label[deforestation_past_years == date] = 1
		ic(np.unique(deforestation_past_years, return_counts=True))
		print("Past deforestation different from 0 (no deforestation)")
		label[np.logical_and(deforestation_past_years < date, 
			deforestation_past_years != 0)] = 2 # includes <=2007 deforestation 
		ic(np.unique(deforestation_past_years, return_counts=True))
		print("Past deforestation before 2008 is 2")
		label_past_deforestation_before_2008 = self.loadPastDeforestationBefore2008()
		ic(np.unique(label_past_deforestation_before_2008, return_counts=True))
		label[label_past_deforestation_before_2008 == 1] = 2
		ic(np.unique(deforestation_past_years, return_counts=True))

		return label

	def getLabelCurrentDeforestation(self, label_mask, selected_class = 1):
		super().getLabelCurrentDeforestation(label_mask[-1], 
			selected_class = selected_class)

	def addCloudMaskToLabel(self, label, date):
		cloud_mask = np.load(self.paths.cloud_mask[date])
		label[cloud_mask == 1] = 2
		return label

	def loadLabelFromProject(self):
		label = np.load('D:/Jorge/datasets/deforestation/Para_2018_2019/mask_label_17730x9203.npy')
		return label
class MT(Dataset): 
    def __init__(self): 
        self.paths = PathsMT() 
 
        self.site = 'MT' 
         
        self.lims = np.array([0, 20795-4000, 0+3000, 13420]) 
 
        self.grid_x, self.grid_y = 5,5 
 
        self.label_filename = 'ref_2019_2020_20798x13420.npy' 
 
    def loadLabel(self): 
        label = np.load(self.paths.label + self.label_filename).astype('uint8')[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] 
 
        label_past_deforestation_before_2008 = utils_v1.load_tiff_image( 
            self.paths.deforestation_before_2008).astype('uint8')[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] 
        label[label_past_deforestation_before_2008 == 1] = 2 
        del label_past_deforestation_before_2008 
        return label 
 
    def loadInputImage(self): 
        return np.load(self.paths.optical_im + 'optical_im.npy').astype('float32')[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] 
     
    def calculateTiles(self): 
        # Defining tiles for training, validation and test sets 
        tiles_tr = [2,4,5,6,7,12,14,15,18,21,23,24]  
        tiles_val = [9,11,25] 
        tiles_ts = list(set(np.arange(self.grid_x * self.grid_y)+1)-set(tiles_tr)-set(tiles_val)) 
        return tiles_tr, tiles_val, tiles_ts 
    def loadPastDeforestationLabel(self): 
        label_past_deforestation = self.loadLabel() 
 
        label_past_deforestation[label_past_deforestation == 1] = 0 
        label_past_deforestation[label_past_deforestation == 2] = 1 
 
        return label_past_deforestation 
 
 
 
class MTDeforestationTime(MT): 
    def __init__(self, addPastDeforestationInput = True): 
        self.addPastDeforestationInput = addPastDeforestationInput 
        super().__init__() 
 
    def loadInputImage(self): 
        image_stack = super().loadInputImage() 
        if self.addPastDeforestationInput == True: 
            image_stack = self.addDeforestationTime(image_stack) 
        # image_stack = self.addPastDeforestation(image_stack) 
        ic(image_stack.shape) 
        return image_stack   
    def addDeforestationTime(self, image_stack): 
        deforestation_time = np.load(self.paths.label + 'deforestation_time_normalized_2019_2020.npy')[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] # has past deforestation up to 2018 
 
        ic(np.unique(deforestation_time, return_counts=True)) 
         
        ic(deforestation_time.shape, image_stack.shape) 
        image_stack = np.concatenate((deforestation_time, image_stack), axis = -1) 
        del deforestation_time   
        return image_stack


class MTMultipleDates(MTDeforestationTime):
	def __init__(self, dates = [2017, 2018, 2019], addPastDeforestationInput = True, borderBuffer = 0):
		super().__init__(addPastDeforestationInput)
		self.dates = dates
		# self.date_ids = [0,1]
		self.date_ids = range(len(dates[:-1]))
		ic(list(self.date_ids))
		self.image_channels = []

		if self.addPastDeforestationInput == True:
			# self.image_channels = [[0,] + list(range(2,22)),
			# 	[1,] + list(range(12,32))]
			for date_id in self.date_ids:
				self.image_channels.append([date_id,] + list(range(date_id * 10 + len(self.date_ids), 
					date_id * 10 + 20 + len(self.date_ids)))) 
			# image_channels_check = [[0,] + list(range(2,22)),
			# 	[1,] + list(range(12,32))]
			# assert image_channels_check == self.image_channels
			
		else:
			for date_id in self.date_ids:
				self.image_channels.append([date_id,] + list(range(date_id * 10, 
					date_id * 10 + 20))) 
		ic(self.image_channels)

		self.borderBuffer = borderBuffer
		# ic(self.image_channels == image_channels_check)
	def loadPastDeforestationBefore2008(self):
		label_past_deforestation_before_2008 = utils_v1.load_tiff_image(
			self.paths.deforestation_before_2008)
		return label_past_deforestation_before_2008
	def loadInputImage(self):
		# self.addPastDeforestationInput = False
		# image_stack = super().loadInputImage()
		# self.addPastDeforestationInput = True
		image_stack = []
		for date in self.dates:
			image_stack.append(
				np.load(self.paths.optical_im_past_dates[date] + 'optical_im.npy').astype('float32')[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]]
			)
		image_stack = np.concatenate(image_stack, axis = -1)
		'''
		image_stack = np.concatenate((
			np.load(self.paths.optical_im_past_dates[2017] + 'optical_im.npy').astype('float32'),
			image_stack),
			axis = -1)
		'''

		if self.addPastDeforestationInput == True:
			image_stack = self.addDeforestationTime(image_stack)        
		ic(image_stack.shape)
		return image_stack

	def addDeforestationTime(self, image_stack):
		'''
		image_stack = super().addDeforestationTime(image_stack)
		print("2")
		ic(image_stack.shape)
		'''
		deforestation_times = []
		for date in self.dates[:-1]:
			deforestation_times.append(
				np.load(self.paths.deforestation_time[date]).astype(np.float32)[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] 
			)
		deforestation_times = np.concatenate(deforestation_times, axis = -1)
		image_stack = np.concatenate((deforestation_times, 
			image_stack), axis = -1)
		del deforestation_times
		
		return image_stack

	def loadLabel(self):
		
		label_per_date = []
		for date in self.dates[1:]:
			label = self.loadLabelFromDate(date)
			
			if self.borderBuffer > 0:
				label = self.removeBorderBufferFromLabel(label, self.borderBuffer)

			label_per_date.append(
				np.expand_dims(label, axis = -1)
			)
		label_per_date = np.concatenate(label_per_date, axis = -1)

		'''
		label_past_date = self.loadLabelFromDate(2018)
		label_current_date = super().loadLabel()

		ic(np.unique(label_current_date, return_counts=True),
			np.unique(label_past_date, return_counts=True))

		label_per_date = np.concatenate((
			np.expand_dims(label_past_date, axis = -1),
			np.expand_dims(label_current_date, axis = -1)),
			axis = -1)
		del label_past_date, label_current_date
		'''
		ic(label_per_date.shape)
		return label_per_date

	def loadLabelFromDate(self, date):
		deforestation_past_years = utils_v1.load_tiff_image(
			self.paths.deforestation_past_years).astype(np.uint8)[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] 
		label = np.zeros_like(deforestation_past_years, dtype = np.uint8)
		print("Loaded deforestation past years")
		ic(np.unique(deforestation_past_years, return_counts=True))
		print("Label where deforestation past years is actual date (2018) = 1")
		label[deforestation_past_years == date] = 1
		ic(np.unique(deforestation_past_years, return_counts=True))
		print("Past deforestation different from 0 (no deforestation)")
		label[np.logical_and(deforestation_past_years < date, 
			deforestation_past_years != 0)] = 2 # includes <=2007 deforestation 
		ic(np.unique(deforestation_past_years, return_counts=True))
		print("Past deforestation before 2008 is 2")
		label_past_deforestation_before_2008 = self.loadPastDeforestationBefore2008().astype(np.uint8)[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3]] 
		ic(np.unique(label_past_deforestation_before_2008, return_counts=True))
		label[label_past_deforestation_before_2008 == 1] = 2
		ic(np.unique(deforestation_past_years, return_counts=True))

		return label

	def getLabelCurrentDeforestation(self, label_mask, selected_class = 1):
		super().getLabelCurrentDeforestation(label_mask[-1], 
			selected_class = selected_class)
