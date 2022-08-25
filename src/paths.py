class Paths():
    pass

class PathsPara(Paths):
    def __init__(self):
        self.optical_im = 'D:/Jorge/datasets/sentinel2/Para_2018_2019/'
        self.label = 'D:/Jorge/datasets/deforestation/Para_2018_2019/'
        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/'

        self.deforestation_time_name = 'deforestation_time_normalized_2018_2019.npy'
        self.deforestation_before_2008 = 'D:/Jorge/datasets/deforestation/deforestation_before_2008/deforestation_before_2008_para.tif'
        self.hydrography = 'D:/Jorge/datasets/deforestation/Para/hydgrography.tif'
        
        self.deforestation_past_years = 'D:/Jorge/datasets/deforestation/Para/deforestation_past_years.tif'

        self.deforestation_time = {
            2019: 'D:/Jorge/datasets/regeneration/Para/deforestation_time_normalized_2019.npy',
            2018: 'D:/Jorge/datasets/regeneration/Para/deforestation_time_normalized_2018.npy',
            2017: 'D:/Jorge/datasets/regeneration/Para/deforestation_time_normalized_2017.npy',
            2016: 'D:/Jorge/datasets/regeneration/Para/deforestation_time_normalized_2016.npy',
            2015: 'D:/Jorge/datasets/regeneration/Para/deforestation_time_normalized_2015.npy',
        } 

        self.distance_map_past_deforestation = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2019/distance_map_past_deforestation.npy'
        self.distance_map_past_deforestation_2016 = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2016/distance_map_past_deforestation_2016.npy'
        self.distance_map_past_deforestation_2017 = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2017/distance_map_past_deforestation_2017.npy'
        self.distance_map_past_deforestation_2018 = 'D:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2018/distance_map_past_deforestation_2018.npy'


        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/'

        self.optical_im_past_dates = {
            2019: self.optical_im_folder + 'Para_2019/',
            2018: self.optical_im_folder + 'Para_2018/',
            2017: self.optical_im_folder + 'Para_2017/',
            2016: self.optical_im_folder + 'Para_2016/',
            2015: self.optical_im_folder + 'Para_2015/'
        }

        self.cloud_mask = {
            2019: self.optical_im_folder + 'Para_2019/' + 'cloudmask_Para_2019.npy',
            2018: self.optical_im_folder + 'Para_2018/' + 'cloudmask_Para_2018.npy',
            2017: self.optical_im_folder + 'Para_2017/' + 'cloudmask_Para_2017.npy',
            2016: self.optical_im_folder + 'Para_2016/' + 'cloudmask_Para_2016.npy',
            2015: self.optical_im_folder + 'Para_2015/' + 'cloudmask_Para_2015.npy'
        }

        self.labelFromProject = 'D:/Jorge/datasets/deforestation/Para_2018_2019/mask_label_17730x9203.npy'
class PathsMT(Paths): 
    def __init__(self): 
        self.optical_im = 'D:/Jorge/datasets/sentinel2/MT_2019_2020/' 
        self.label = 'D:/Jorge/datasets/deforestation/MT_2019_2020/' 
        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/MT/' 

        self.deforestation_time_name = 'deforestation_time_normalized_2019_2020.npy'

        self.deforestation_before_2008 = 'D:/Jorge/datasets/deforestation/MT/deforestation_before_2008/deforestation_before_2008_MT.tif' 
        self.hydrography = 'D:/Jorge/datasets/deforestation/MT/hydgrography.tif'
        self.deforestation_past_years = 'D:/Jorge/datasets/deforestation/MT/deforestation_past_years.tif' 

        self.deforestation_time = { 
            2019: 'D:/Jorge/datasets/regeneration/MT/deforestation_time_normalized_2019.npy', 
            2018: 'D:/Jorge/datasets/regeneration/MT/deforestation_time_normalized_2018.npy', 
            2017: 'D:/Jorge/datasets/regeneration/MT/deforestation_time_normalized_2017.npy', 
            2016: 'D:/Jorge/datasets/regeneration/MT/deforestation_time_normalized_2016.npy' 
        }  
 
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/'

        self.optical_im_past_dates = { 
            2020: self.optical_im_folder + 'MT_2020/', 
            2019: self.optical_im_folder + 'MT_2019/',  
            2018: self.optical_im_folder + 'MT_2018/', 
            2017: self.optical_im_folder + 'MT_2017/', 
            2016: self.optical_im_folder + 'MT_2016/' 
        }

        self.cloud_mask = {
            2020: self.optical_im_folder + 'MT_2020/' + 'cloudmask_MT_2020.npy',
            2019: self.optical_im_folder + 'MT_2019/' + 'cloudmask_MT_2019.npy',
            2018: self.optical_im_folder + 'MT_2018/' + 'cloudmask_MT_2018.npy',
            2017: self.optical_im_folder + 'MT_2017/' + 'cloudmask_MT_2017.npy',
            2016: self.optical_im_folder + 'MT_2016/' + 'cloudmask_MT_2016.npy'
        }

        self.labelFromProject = 'D:/Jorge/datasets/deforestation/MT_2019_2020/ref_2019_2020_20798x13420.npy'
