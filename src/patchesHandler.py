import numpy as np
from icecream import ic
import skimage
import pdb
import tensorflow as tf

class PatchesHandler():

	def create_idx_image(self, ref_mask):
		h = ref_mask.shape[0]
		w = ref_mask.shape[1]
		
		im_idx_row = np.repeat(
			np.expand_dims(np.arange(ref_mask.shape[0], dtype = np.uint16), 
					axis = -1),
			w, axis = -1).astype(np.uint16)
		im_idx_col = np.repeat(
			np.expand_dims(np.arange(ref_mask.shape[1], dtype = np.uint16),
					axis = 0),
			h, axis = 0).astype(np.uint16)
		
		im_idx_row = np.expand_dims(im_idx_row, axis = -1)
		im_idx_col = np.expand_dims(im_idx_col, axis = -1)
		ic(im_idx_row.shape, im_idx_col.shape)
		ic(im_idx_row.dtype, im_idx_col.dtype)

		im_idx = np.concatenate(
				(im_idx_row,im_idx_col),
				axis = -1)
		del im_idx_row, im_idx_col
		ic(im_idx.shape, im_idx.dtype)
		return im_idx


	def extract_patches(self, im_idx, patch_size, overlap):
		'''
		overlap range: 0 - 1 

		coords: all the coords available in the image for patch_size and overlap
		'''
		row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
		coords = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps, 2)).astype(np.uint16)
		ic(coords.shape, coords.dtype)

		coords = coords.reshape(-1, patch_size[0], patch_size[1], 2)
		coords = np.squeeze(coords[:, 0, 0, :])
		ic(coords.shape, coords.dtype)
		return coords.astype(np.uint16)
	
	def trainTestSplit(self, coords, mask_tr_val, patch_size):
		coords_train, coords_val = [], []
		for idx in range(coords.shape[0]):
			mask_patch = mask_tr_val[coords[idx, 0]:coords[idx, 0] + patch_size[0], 
				coords[idx, 1]:coords[idx, 1] + patch_size[1]]
			# ic(mask_patch.shape)
			# pdb.set_trace()
			if np.all(mask_patch == 1):
				coords_train.append(coords[idx])
			elif np.all(mask_patch == 2):
				coords_val.append(coords[idx])
		del coords
		coords_train = np.array(coords_train, dtype = np.uint16)
		coords_val = np.array(coords_val, dtype = np.uint16)

		return coords_train, coords_val


	def retrieve_idx_percentage(self, reference, coords, patch_size, pertentage = 5):
		coords_to_return = []
		for idx in range(coords.shape[0]):
			patch_ref = reference[coords[idx,0]:coords[idx,0]+patch_size,
					coords[idx,1]:coords[idx,1]+patch_size]
			class1 = patch_ref[patch_ref==1]
			if len(class1) >= int((patch_size**2)*(pertentage/100)):
				coords_to_return.append(coords[idx])
		del coords, reference
		return np.asarray(coords_to_return, dtype=np.uint16)


	def batch_generator(self, batches, image, reference, patch_size, number_class):
		"""Take as input a Keras ImageGen (Iterator) and generate random
		crops from the image batches generated by the original iterator.
		"""

		while True:
			batch_coords, _ = next(batches)
			batch_coords = np.squeeze(batch_coords.astype(np.uint16))
			
			batch_img = np.zeros((batch_coords.shape[0], patch_size, patch_size, image.shape[-1]), dtype = np.float32)
			batch_ref = np.zeros((batch_coords.shape[0], patch_size, patch_size, number_class), dtype = np.float32)

			for i in range(batch_coords.shape[0]):
				batch_img[i] = image[batch_coords[i,0] : batch_coords[i,0] + patch_size,
						batch_coords[i,1] : batch_coords[i,1] + patch_size] 
				batch_ref_int = reference[batch_coords[i,0] : batch_coords[i,0] + patch_size,
						batch_coords[i,1] : batch_coords[i,1] + patch_size]

				if np.random.rand()<0.3:
					batch_img[i] = np.rot90(batch_img[i], 1)
					batch_ref_int = np.rot90(batch_ref_int, 1)
					
				if np.random.rand() >= 0.3 and np.random.rand() <= 0.5:
					batch_img[i] = np.flip(batch_img[i], 0)
					batch_ref_int = np.flip(batch_ref_int, 0)
				
				if np.random.rand() > 0.5 and np.random.rand() <= 0.7:
					batch_img[i] = np.flip(batch_img[i], 1)
					batch_ref_int = np.flip(batch_ref_int, 1)
					
				if np.random.rand() > 0.7:
					batch_img[i] = batch_img[i]
					batch_ref_int = batch_ref_int
				batch_ref[i] = tf.keras.utils.to_categorical(batch_ref_int, number_class)
			yield (batch_img, batch_ref)

	def infer(self, new_model, image1_pad,
		h, w, num_patches_x, num_patches_y, 
		patch_size_x, patch_size_y):
		img_reconstructed = np.zeros((h, w), dtype=np.float32)
		for i in range(0,num_patches_y):
			for j in range(0,num_patches_x):
				patch = image1_pad[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]
				predicted = new_model.predict(np.expand_dims(patch, axis=0))[:,:,:,1].astype(np.float32)
				img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)] = predicted
		del patch, predicted
		return img_reconstructed

class PatchesHandlerMultipleDates(PatchesHandler):
	def __init__(self, addPastDeforestationInput):
		if addPastDeforestationInput == True:
			self.image_channels = [[0,] + list(range(2,22)),
				[1,] + list(range(12,32))]
			self.input_image_shape = 21
		else:
			self.image_channels = [list(range(0,20)),
				list(range(10,30))]
			self.input_image_shape = 20
		self.input_image_shape = len(self.image_channels[0])
		ic(self.input_image_shape)
		ic(self.image_channels)
	'''
	def infer(self, new_model, image1_pad, h, w, 
                    num_patches_x, num_patches_y, patch_size_rows, 
                    patch_size_cols):
		image1_pad = image1_pad[..., self.image_channels[-1]]
		super().infer(new_model, image1_pad, h, w, 
                    num_patches_x, num_patches_y, patch_size_rows, 
                    patch_size_cols)
	'''
	def trainTestSplit(self, coords, mask_tr_val, patch_size):
		coords_current_date_train, coords_current_date_val = super().trainTestSplit(
			coords, mask_tr_val, patch_size)

		mask_past_date_tr_val = np.ones_like(mask_tr_val)

		# mask_past_date_tr_val = mask_tr_val.copy()
		# mask_past_date_tr_val[mask_past_date_tr_val == 0] = 1
		
		coords_past_date_train, _ = super().trainTestSplit(
			coords, mask_past_date_tr_val, patch_size)
		
		ic(coords_current_date_train.shape, coords_past_date_train.shape)
		ic(coords_current_date_train[20])
		coords_current_date_train = self.addDateToCoords(
			coords_current_date_train, 1)
		coords_current_date_val = self.addDateToCoords(
			coords_current_date_val, 1)
		coords_past_date_train = self.addDateToCoords(
			coords_past_date_train, 0)

		ic(coords_current_date_train[20])

		ic(coords_current_date_train.shape, coords_past_date_train.shape)
		# pdb.set_trace()
		# return np.concatenate((coords_current_date_train, 
		#	coords_past_date_train), axis = 0), coords_current_date_val
		return coords_current_date_train, coords_current_date_val
		
			# np.concatenate((coords_current_date_val, 
			# coords_past_date_val), axis = 0)

	def addDateToCoords(self, coords, date):
		return np.insert(coords, coords.shape[1], date, axis = 1)

	def retrieve_idx_percentage(self, reference, coords, patch_size, pertentage = 5):
		coords_to_return = []
		for idx in range(coords.shape[0]):
			patch_ref = reference[coords[idx,0]:coords[idx,0]+patch_size,
					coords[idx,1]:coords[idx,1]+patch_size, coords[idx,2]]
			class1 = patch_ref[patch_ref==1]
			if len(class1) >= int((patch_size**2)*(pertentage/100)):
				coords_to_return.append(coords[idx])
		del coords, reference
		return np.asarray(coords_to_return, dtype=np.uint16)

	def batch_generator(self, batches, image, reference, patch_size, number_class):
		"""Take as input a Keras ImageGen (Iterator) and generate random
		crops from the image batches generated by the original iterator.
		"""
			

		while True:
			batch_coords, _ = next(batches)
			batch_coords = np.squeeze(batch_coords.astype(np.uint16))
			
			batch_img = np.zeros((batch_coords.shape[0], patch_size, patch_size, self.input_image_shape), dtype = np.float32)
			batch_ref = np.zeros((batch_coords.shape[0], patch_size, patch_size, number_class), dtype = np.float32)
			
			for i in range(batch_coords.shape[0]):
				# print("image.shape, batch_coords.shape, batch_coords",
				# 	image.shape, batch_coords.shape, batch_coords)
				# raise KeyboardInterrupt
				try:
					batch_ref_int = reference[batch_coords[i,0] : batch_coords[i,0] + patch_size,
							batch_coords[i,1] : batch_coords[i,1] + patch_size, batch_coords[i,2]]
					batch_img[i] = image[batch_coords[i,0] : batch_coords[i,0] + patch_size,
							batch_coords[i,1] : batch_coords[i,1] + patch_size, self.image_channels[batch_coords[i,2]]] 
				except:
					# print("batch_coords.shape, batch_coords[i], batch_coords[i].shape",
					# 	batch_coords.shape, batch_coords[i], batch_coords[i].shape)
					pass
				if np.random.rand()<0.3:
					batch_img[i] = np.rot90(batch_img[i], 1)
					batch_ref_int = np.rot90(batch_ref_int, 1)
					
				if np.random.rand() >= 0.3 and np.random.rand() <= 0.5:
					batch_img[i] = np.flip(batch_img[i], 0)
					batch_ref_int = np.flip(batch_ref_int, 0)
				
				if np.random.rand() > 0.5 and np.random.rand() <= 0.7:
					batch_img[i] = np.flip(batch_img[i], 1)
					batch_ref_int = np.flip(batch_ref_int, 1)
					
				if np.random.rand() > 0.7:
					batch_img[i] = batch_img[i]
					batch_ref_int = batch_ref_int
				batch_ref[i] = tf.keras.utils.to_categorical(batch_ref_int, number_class)
			yield (batch_img, batch_ref)

'''
def create_idx_image(ref_mask):
	im_idx = np.arange(ref_mask.shape[0] * ref_mask.shape[1]).reshape(ref_mask.shape[0] , ref_mask.shape[1])
	return im_idx

def retrieve_idx_percentage(reference, patches_idx_set, patch_size, pertentage = 5):
	count = 0
	new_idx_patches = []
	reference_vec = reference.reshape(reference.shape[0]*reference.shape[1])
	for patchs_idx in patches_idx_set:
		patch_ref = reference_vec[patchs_idx]
		class1 = patch_ref[patch_ref==1]
		if len(class1) >= int((patch_size**2)*(pertentage/100)):
			count = count + 1
			new_idx_patches.append(patchs_idx)
	return np.asarray(new_idx_patches)
def extract_patches2(im_idx, patch_size, overlap):
	# overlap range: 0 - 1 
	row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
	patches = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps, 2))
	return patches

'''