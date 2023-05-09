import numpy as np
import utils_v1
class LandsatLoader():
    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):
        ims = []
        for path in self.dataset.paths.landsat:
            ims.append(utils_v1.load_tiff_image(path))
        return ims


