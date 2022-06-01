class Paths():
    pass

class PathsPara(Paths):
    def __init__(self):
        self.optical_im = 'D:/Jorge/datasets/sentinel2/Para_2018_2019/'
        self.label = 'D:/Jorge/datasets/deforestation/Para_2018_2019/'
        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/'

        self.deforestation_before_2008 = 'D:/Jorge/datasets/deforestation/deforestation_before_2008/deforestation_before_2008_para.tif'

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


        self.optical_im_past_dates = {
            2019: 'D:/Jorge/datasets/sentinel2/Para_2019/',
            2018: 'D:/Jorge/datasets/sentinel2/Para_2018/',
            2017: 'D:/Jorge/datasets/sentinel2/Para_2017/',
            2016: 'D:/Jorge/datasets/sentinel2/Para_2016/',
            2015: 'D:/Jorge/datasets/sentinel2/Para_2015/'
        }
