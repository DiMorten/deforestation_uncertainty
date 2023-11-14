import numpy as np
import utils_v1
import cv2
import skimage
from skimage.exposure import match_histograms

class LandsatLoader():
	def __init__(self, dataset, im_shape):
		self.dataset = dataset
		self.im_shape = im_shape

	def load(self):
		ims = []
		for path in self.dataset.paths.landsat:
			print("Loading {}".format(path))
			im = utils_v1.load_tiff_image(path)
			# im = (im*0.0001*256).astype(np.uint8)
			im = ((im*2.75e-05-0.2)*256).astype(np.uint8)
			
			im = np.transpose(im, (1, 2, 0))[...,0:3]
			# im = im[...,0:3]
			im = im[self.dataset.lims[0]:self.dataset.lims[1], self.dataset.lims[2]:self.dataset.lims[3]] 
			ims.append(im)
		return ims
	def darken_past_deforestation(self, ims, label):
		
		for im in ims:
			im[label == 2] //= 10
		return ims

	def get_borders_from_label(self, label, borderBuffer):
		# Creation of border buffer for pixels not considered
		label[label == 2] = 0
		im_dilate = skimage.morphology.dilation(label, skimage.morphology.disk(borderBuffer))
		im_erosion = skimage.morphology.erosion(label, skimage.morphology.disk(borderBuffer))
		inner_buffer = label - im_erosion
		inner_buffer[inner_buffer == 1] = 1
		outer_buffer = im_dilate-label
		outer_buffer[outer_buffer == 1] = 1
		
		buffer = inner_buffer + outer_buffer

		return buffer   

	def add_deforestation_edges_by_date(self, ims, label_dates):
		color = [255, 255, 0]
		for idx in range(len(label_dates)):
			label_dates[idx] = self.get_borders_from_label(label_dates[idx], borderBuffer=2)
		for idx, (im, label_date) in enumerate(zip(ims, label_dates)):
			if idx != 0:
				for chan in range(3):
					im[...,chan][label_date == 1] = color[chan]
		return ims


	def add_deforestation_edges_by_date(self, ims, label_dates):
		color = {1: [255, 0, 0], # [255, 165, 0]
				2: [255, 255, 0]} # [255, 0, 0]
		for idx in range(len(label_dates)):
			label_dates[idx] = self.get_borders_from_label(label_dates[idx], borderBuffer=3)


		for chan in range(3):
			ims[2][...,chan][label_dates[1] == 1] = color[1][chan]
		for idx, (im, label_date) in enumerate(zip(ims, label_dates)):
			if idx != 0:
				for chan in range(3):
					im[...,chan][label_date == 1] = color[idx][chan]

		return ims

class LandsatLoaderHistogramMatching(LandsatLoader):
	def load(self):
		
		im_histogram_matching = utils_v1.load_tiff_image(self.dataset.paths.landsat_matching)
		im_histogram_matching = np.transpose(im_histogram_matching, (1, 2, 0))[...,0:3]
		im_histogram_matching = im_histogram_matching[self.dataset.lims[0]:self.dataset.lims[1], self.dataset.lims[2]:self.dataset.lims[3]] 
		# resize im_histogram_matching
		# print("self.im_shape", self.im_shape)
		im_histogram_matching = cv2.resize(im_histogram_matching, (self.im_shape[1], self.im_shape[0]))
		# print("im_histogram_matching.shape", im_histogram_matching.shape)
		ims = []
		for path in self.dataset.paths.landsat:
			print("Loading {}".format(path))
			im = utils_v1.load_tiff_image(path)
			im = ((im*2.75e-05-0.2)*256).astype(np.uint8)
			im = np.transpose(im, (1, 2, 0))[...,0:3]
			im = im[self.dataset.lims[0]:self.dataset.lims[1], self.dataset.lims[2]:self.dataset.lims[3]] 
			im = match_histograms(im, im_histogram_matching, channel_axis=-1)
			ims.append(im)
			# ims.append(im)
		return ims