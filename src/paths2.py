class Paths():
    pass

class PathsPara(Paths):
    def __init__(self):
        self.optical_im = 'E:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/Para_10m/Sentinel2_2018/'
        self.label = 'E:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/Para_10m/'
        self.experiment = 'E:/jorg/phd/datasets/deforestation/experiments/'

        self.deforestation_time_name = 'deforestation_time_normalized_2018_2019.npy'
        self.deforestation_before_2008 = 'E:/jorg/phd/datasets/deforestation/deforestation_before_2008/deforestation_before_2008_para.tif'

        self.deforestation_past_years = 'E:/jorg/phd/datasets/deforestation/Para/deforestation_past_years.tif'

        self.optical_im_folder = 'E:/jorg/phd/datasets/sentinel2/'

        self.deforestation_time = {
            2019: 'E:/jorg/phd/datasets/regeneration/Para/deforestation_time_normalized_2019.npy',
            2018: 'E:/jorg/phd/datasets/regeneration/Para/deforestation_time_normalized_2018.npy',
            2017: 'E:/jorg/phd/datasets/regeneration/Para/deforestation_time_normalized_2017.npy',
            2016: 'E:/jorg/phd/datasets/regeneration/Para/deforestation_time_normalized_2016.npy',
            2015: 'E:/jorg/phd/datasets/regeneration/Para/deforestation_time_normalized_2015.npy',
        } 

        self.distance_map_past_deforestation = 'E:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2019/distance_map_past_deforestation.npy'
        self.distance_map_past_deforestation_2016 = 'E:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2016/distance_map_past_deforestation_2016.npy'
        self.distance_map_past_deforestation_2017 = 'E:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2017/distance_map_past_deforestation_2017.npy'
        self.distance_map_past_deforestation_2018 = 'E:/jorg/phd/fifth_semester/project_forestcare/dataset_deforestation/Para_2018/distance_map_past_deforestation_2018.npy'



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
        self.experiment = 'E:/jorg/phd/datasets/deforestation/experiments/MT/' 

        self.deforestation_time_name = 'deforestation_time_normalized_2019_2020.npy'

        self.deforestation_before_2008 = 'E:/jorg/phd/datasets/deforestation/MT/deforestation_before_2008/deforestation_before_2008_MT.tif' 
 
        self.deforestation_past_years = 'E:/jorg/phd/datasets/deforestation/MT/deforestation_past_years.tif' 



        self.deforestation_time = {
            2019: 'E:/jorg/phd/datasets/regeneration/MT/deforestation_time_normalized_2019.npy',
            2018: 'E:/jorg/phd/datasets/regeneration/MT/deforestation_time_normalized_2018.npy',
            2017: 'E:/jorg/phd/datasets/regeneration/MT/deforestation_time_normalized_2017.npy',
            2016: 'E:/jorg/phd/datasets/regeneration/MT/deforestation_time_normalized_2016.npy',
            2015: 'E:/jorg/phd/datasets/regeneration/MT/deforestation_time_normalized_2015.npy',
        } 
        

        self.optical_im_folder = 'E:/jorg/phd/datasets/sentinel2/'

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
