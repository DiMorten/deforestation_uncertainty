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

        self.im_filenames = {
            2020: ['S2_PA_2020_07_15_B1_B2_B3.tif',
        'S2_PA_2020_07_15_B4_B5_B6.tif',
        'S2_PA_2020_07_15_B7_B8_B8A.tif',
        'S2_PA_2020_07_15_B9_B10_B11.tif',
        'S2_PA_2020_07_15_B12.tif'],
            2019: ['COPERNICUS_S2_20190721_20190726_B1_B2_B3.tif',
        'COPERNICUS_S2_20190721_20190726_B4_B5_B6.tif',
        'COPERNICUS_S2_20190721_20190726_B7_B8_B8A.tif',
        'COPERNICUS_S2_20190721_20190726_B9_B10_B11.tif',
        'COPERNICUS_S2_20190721_20190726_B12.tif'],
            2018: ['COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif',
        'COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif',
        'COPERNICUS_S2_20180721_20180726_B7_B8_B8A.tif',
        'COPERNICUS_S2_20180721_20180726_B9_B10_B11.tif',
        'COPERNICUS_S2_20180721_20180726_B12.tif'],

            2015: ['PA_S2_2015_B1_B2_B3_crop.tif', 
            'PA_S2_2015_B4_B5_B6_crop.tif', 
            'PA_S2_2015_B7_B8_B8A_crop.tif', 
            'PA_S2_2015_B9_B10_B11_crop.tif', 
            'PA_S2_2015_B12_crop.tif'] 
        }
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

        self.im_filenames = {
            2020: ['S2_R1_MT_2020_08_03_2020_08_15_B1_B2.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B3_B4.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B5_B6.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B7_B8.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B8A_B9.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B10_B11.tif',
        'S2_R1_MT_2020_08_03_2020_08_15_B12.tif'],
            2019: ['S2_R1_MT_2019_08_02_2019_08_05_B1_B2.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B3_B4.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B5_B6.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B7_B8.tif', 
            'S2_R1_MT_2019_08_02_2019_08_05_B8A_B9.tif',
            'S2_R1_MT_2019_08_02_2019_08_05_B10_B11.tif',
            'S2_R1_MT_2019_08_02_2019_08_05_B12.tif'],
            2018: ['MT_S2_07_26_28_31_2018_B1_B2_crop.tif', 
            'MT_S2_07_26_28_31_2018_B3_B4_crop.tif', 
            'MT_S2_07_26_28_31_2018_B5_B6_crop.tif', 
            'MT_S2_07_26_28_31_2018_B7_B8_crop.tif', 
            'MT_S2_07_26_28_31_2018_B8A_B9_crop.tif',
            'MT_S2_07_26_28_31_2018_B10_B11_crop.tif',
            'MT_S2_07_26_28_31_2018_B12_crop.tif'],
            2017: ['MT_S2_07_26_28_2017_B1_B2_crop.tif', 
            'MT_S2_07_26_28_2017_B3_B4_crop.tif', 
            'MT_S2_07_26_28_2017_B5_B6_crop.tif', 
            'MT_S2_07_26_28_2017_B7_B8_crop.tif', 
            'MT_S2_07_26_28_2017_B8A_B9_crop.tif',
            'MT_S2_07_26_28_2017_B10_B11_crop.tif',
            'MT_S2_07_26_28_2017_B12_crop.tif'],
            2016: ['MT_S2_2016_07_21_08_07_B1_B2_crop.tif', 
            'MT_S2_2016_07_21_08_07_B3_B4_crop.tif', 
            'MT_S2_2016_07_21_08_07_B5_B6_crop.tif', 
            'MT_S2_2016_07_21_08_07_B7_B8_crop.tif', 
            'MT_S2_2016_07_21_08_07_B8A_B9_crop.tif',
            'MT_S2_2016_07_21_08_07_B10_B11_crop.tif',
            'MT_S2_2016_07_21_08_07_B12_crop.tif'],

        }
class PathsMA(Paths): 
    def __init__(self): 

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/MA/' 


        self.hydrography = 'D:/Jorge/datasets/deforestation/MA/hydrography.tif'
        self.deforestation_past_years = 'D:/Jorge/datasets/deforestation/MA/deforestation_past_years.tif' 
 
        self.deforestation_time = {
            2021: 'D:/Jorge/datasets/deforestation/MA/deforestation_time_normalized_2021.npy',
            2020: 'D:/Jorge/datasets/deforestation/MA/deforestation_time_normalized_2020.npy'
        } 

        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/'

        self.optical_im_past_dates = { 
            2021: self.optical_im_folder + 'MA_2021/', 
            2020: self.optical_im_folder + 'MA_2020/',  
        }

        self.cloud_mask = {
            2021: self.optical_im_folder + 'MA_2021/' + 'cloudmask_MA_2021.npy',
            2020: self.optical_im_folder + 'MA_2020/' + 'cloudmask_MA_2020.npy',
        }

        self.im_filenames = {
			2020: ['T23KQV_20200602T130251_B01.jp2', 
            'T23KQV_20200602T130251_B02.jp2', 
            'T23KQV_20200602T130251_B03.jp2', 
            'T23KQV_20200602T130251_B04.jp2', 
            'T23KQV_20200602T130251_B05.jp2',
            'T23KQV_20200602T130251_B06.jp2',
            'T23KQV_20200602T130251_B07.jp2',
            'T23KQV_20200602T130251_B08.jp2',
            'T23KQV_20200602T130251_B8A.jp2',
            'T23KQV_20200602T130251_B09.jp2',
            'T23KQV_20200602T130251_B10.jp2',
            'T23KQV_20200602T130251_B11.jp2',
            'T23KQV_20200602T130251_B12.jp2'],
			2021: ['T23KQV_20210811T130249_B01.jp2', 
            'T23KQV_20210811T130249_B02.jp2', 
            'T23KQV_20210811T130249_B03.jp2', 
            'T23KQV_20210811T130249_B04.jp2', 
            'T23KQV_20210811T130249_B05.jp2',
            'T23KQV_20210811T130249_B06.jp2',
            'T23KQV_20210811T130249_B07.jp2',
            'T23KQV_20210811T130249_B08.jp2',
            'T23KQV_20210811T130249_B8A.jp2',
            'T23KQV_20210811T130249_B09.jp2',
            'T23KQV_20210811T130249_B10.jp2',
            'T23KQV_20210811T130249_B11.jp2',
            'T23KQV_20210811T130249_B12.jp2'] 
		}