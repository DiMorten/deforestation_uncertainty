import os
class Paths():
    pass

class PathsPA(Paths):
    def __init__(self):
        self.reference_folder = 'D:/Jorge/datasets/deforestation/PA/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/PA/'
        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/PA/'

        self.optical_im = os.path.join(self.optical_im_folder,  '2019')

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2008_PA.tif')

        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')

        self.deforestation_time = {
            2019: os.path.join(self.reference_folder, 'deforestation_time_normalized_2019.npy'),
            2018: os.path.join(self.reference_folder, 'deforestation_time_normalized_2018.npy'),
            2017: os.path.join(self.reference_folder, 'deforestation_time_normalized_2017.npy'),
            2016: os.path.join(self.reference_folder, 'deforestation_time_normalized_2016.npy'),
            2015: os.path.join(self.reference_folder, 'deforestation_time_normalized_2015.npy'),
        } 

        self.optical_im_past_dates = {
            2019: os.path.join(self.optical_im_folder,  '2019'),
            2018: os.path.join(self.optical_im_folder,  '2018'),
            2017: os.path.join(self.optical_im_folder,  '2017'),
            2016: os.path.join(self.optical_im_folder,  '2016'),
            2015: os.path.join(self.optical_im_folder,  '2015')
        }

        self.cloud_mask = {
            2019: os.path.join(self.optical_im_folder,  '2019', 'cloudmask_PA_2019.npy'),
            2018: os.path.join(self.optical_im_folder,  '2018', 'cloudmask_PA_2018.npy'),
            2017: os.path.join(self.optical_im_folder,  '2017', 'cloudmask_PA_2017.npy'),
            2016: os.path.join(self.optical_im_folder,  '2016', 'cloudmask_PA_2016.npy'),
            2015: os.path.join(self.optical_im_folder,  '2015', 'cloudmask_PA_2015.npy'),
        }


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

        landsat_base = 'D:/Jorge/datasets/landsat/PA/'
        self.landsat = [
            os.path.join(landsat_base, 'landsat_PA_2018.tif'),
            os.path.join(landsat_base, 'landsat_PA_2019.tif'),
            os.path.join(landsat_base, 'landsat_PA_2020.tif'),

        ]
class PathsMT(Paths): 
    def __init__(self): 
        self.reference_folder = 'D:/Jorge/datasets/deforestation/MT/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/MT/'
        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/MT/' 

        self.optical_im = os.path.join(self.optical_im_folder,  '2020')

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2008_MT.tif') 
        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif') 

        self.deforestation_time = { 
            2019: os.path.join(self.reference_folder, 'deforestation_time_normalized_2019.npy'),
            2018: os.path.join(self.reference_folder, 'deforestation_time_normalized_2018.npy'),
            2017: os.path.join(self.reference_folder, 'deforestation_time_normalized_2017.npy'),
            2016: os.path.join(self.reference_folder, 'deforestation_time_normalized_2016.npy'),
        }  
 
        self.optical_im_past_dates = { 
            2020: os.path.join(self.optical_im_folder,  '2020'),
            2019: os.path.join(self.optical_im_folder,  '2019'),
            2018: os.path.join(self.optical_im_folder,  '2018'),
            2017: os.path.join(self.optical_im_folder,  '2017'),
            2016: os.path.join(self.optical_im_folder,  '2016')
        }

        self.cloud_mask = {
            2020: os.path.join(self.optical_im_folder,  '2020', 'cloudmask_MT_2020.npy'),
            2019: os.path.join(self.optical_im_folder,  '2019', 'cloudmask_MT_2019.npy'),
            2018: os.path.join(self.optical_im_folder,  '2018', 'cloudmask_MT_2018.npy'),
            2017: os.path.join(self.optical_im_folder,  '2017', 'cloudmask_MT_2017.npy'),
            2016: os.path.join(self.optical_im_folder,  '2016', 'cloudmask_MT_2016.npy'),
        }


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
        landsat_base = 'D:/Jorge/datasets/landsat/MT/'
        self.landsat = [
            os.path.join(landsat_base, 'landsat_MT_2019.tif'),
            os.path.join(landsat_base, 'landsat_MT_2020.tif'),
            os.path.join(landsat_base, 'landsat_MT_2021.tif'),

        ]

class PathsMS(Paths): 
    def __init__(self): 
        self.reference_folder = 'D:/Jorge/datasets/deforestation/MS/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/MS/'

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/MS/' 

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2000_MS.tif') 

        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')
 
        self.deforestation_time = {
            2021: os.path.join(self.reference_folder, 'deforestation_time_normalized_2021.npy'),
            2020: os.path.join(self.reference_folder, 'deforestation_time_normalized_2020.npy'),
            2019: os.path.join(self.reference_folder, 'deforestation_time_normalized_2019.npy'),
            2018: os.path.join(self.reference_folder, 'deforestation_time_normalized_2018.npy'),
        } 

        self.optical_im_past_dates = { 
            2021: os.path.join(self.optical_im_folder,  '2021'), 
            2020: os.path.join(self.optical_im_folder,  '2020'), 
            2019: os.path.join(self.optical_im_folder,  '2019'), 
            2018: os.path.join(self.optical_im_folder,  '2018'), 
        }

        self.cloud_mask = {
            2021: os.path.join(self.optical_im_folder,  '2021', 'cloudmask_MS_2021.npy'), 
            2020: os.path.join(self.optical_im_folder,  '2020', 'cloudmask_MS_2020.npy'), 
            2019: os.path.join(self.optical_im_folder,  '2019', 'cloudmask_MS_2019.npy'),
            2018: os.path.join(self.optical_im_folder,  '2018', 'cloudmask_MS_2018.npy'),
        }
        
        self.im_filenames = {
            2018: ['S2_MS_B1_B2_2018_crop.tif',
                   'S2_MS_B3_B4_2018_crop.tif',
                   'S2_MS_B5_B6_2018_crop.tif',
                   'S2_MS_B7_B8_2018_crop.tif',
                   'S2_MS_B8A_B9_2018_crop.tif',
                   'S2_MS_B10_B11_2018_crop.tif',
                   'S2_MS_B12_2018_crop.tif'],
            2019: ['S2_MS_B1_B2_2019_crop.tif',
                   'S2_MS_B3_B4_2019_crop.tif',
                   'S2_MS_B5_B6_2019_crop.tif',
                   'S2_MS_B7_B8_2019_crop.tif',
                   'S2_MS_B8A_B9_2019_crop.tif',
                   'S2_MS_B10_B11_2019_crop.tif',
                   'S2_MS_B12_2019_crop.tif'],
            2020: ['S2_MS_B1_B2_2020_crop.tif',
                   'S2_MS_B3_B4_2020_crop.tif',
                   'S2_MS_B5_B6_2020_crop.tif',
                   'S2_MS_B7_B8_2020_crop.tif',
                   'S2_MS_B8A_B9_2020_crop.tif',
                   'S2_MS_B10_B11_2020_crop.tif',
                   'S2_MS_B12_2020_crop.tif']
        }
 
        self.im_filenames = {
            2018: ['S2_MS_B4_B3_2018_crop.tif',
                   'S2_MS_B2_B8_2018_crop.tif'],
            2019: ['S2_MS_B4_B3_2019_crop.tif',
                   'S2_MS_B2_B8_2019_crop.tif'],
            2020: ['S2_MS_B4_B3_2020_crop.tif',
                   'S2_MS_B2_B8_2020_crop.tif']                   
        }   

        self.im_filenames = {
            2018: ['S2_MS_B4_B3_2018_crop.tif',
                   'S2_MS_B2_B8_2018_crop.tif'],
            2019: ['S2_MS_B4_B3_2019_crop.tif',
                   'S2_MS_B2_B8_2019_crop.tif'],
            2020: ['S2_MS_B4_B3_2020_crop.tif',
                   'S2_MS_B2_B8_2020_crop.tif'],  
            2021: ['S2_MS_B4_B3_2021_crop.tif',
                   'S2_MS_B2_B8_2021_crop.tif']  

        }   



        landsat_base = 'D:/Jorge/datasets/landsat/MS/'
        self.landsat = [
            os.path.join(landsat_base, 'L8_MS_2019.tif'),
            os.path.join(landsat_base, 'L8_MS_2020.tif'),
            os.path.join(landsat_base, 'L8_MS_2021.tif'),

        ]

        self.landsat_matching = 'D:/Jorge/datasets/landsat/matching/Cerrado_224073_2019/Landsat8_OLI_224073_20190719_8bits_654_contraste.tif'
        '''
        self.im_filenames = {
            2018: ['merged_2018_crop.tif'],
            2019: ['merged_2019_crop.tif'],
            2020: ['merged_2020_crop.tif'],
            2021: ['S2_MS_B4_B3_2021_crop.tif',
                   'S2_MS_B2_B8_2021_crop.tif']
        }    
        '''
class PathsPI(Paths):
    def __init__(self): 

        self.reference_folder = 'D:/Jorge/datasets/deforestation/PI/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/PI/'

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/PI/' 

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2000_PI.tif') 

        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')
 
        self.deforestation_time = {
            2020: os.path.join(self.reference_folder, 'deforestation_time_normalized_2020.npy'),
            2019: os.path.join(self.reference_folder, 'deforestation_time_normalized_2019.npy'),
            2018: os.path.join(self.reference_folder, 'deforestation_time_normalized_2018.npy'),
            2017: os.path.join(self.reference_folder, 'deforestation_time_normalized_2017.npy'),

        } 

        self.optical_im_past_dates = { 
            2021: os.path.join(self.optical_im_folder,  '2021'), 
            2020: os.path.join(self.optical_im_folder,  '2020'), 
            2019: os.path.join(self.optical_im_folder,  '2019'),
            2018: os.path.join(self.optical_im_folder,  '2018'),
            2017: os.path.join(self.optical_im_folder,  '2017'),
        }

        self.cloud_mask = {
            2020: os.path.join(self.optical_im_folder,  '2020', 'cloudmask_PI_2020.npy'), 
            2019: os.path.join(self.optical_im_folder,  '2019', 'cloudmask_PI_2019.npy'),
            2018: os.path.join(self.optical_im_folder,  '2018', 'cloudmask_PI_2018.npy'),
            2017: os.path.join(self.optical_im_folder,  '2017', 'cloudmask_PI_2017.npy')
        }

        self.im_filenames = {
            2017: ['S2_PI_B4_B3_2017_crop.tif',
                   'S2_PI_B2_B8_2017_crop.tif'],
            2018: ['merged_2019_crop.tif'],
            2019: ['merged_2020_crop.tif'],
            2020: ['merged_2021_crop.tif']
        }
        '''
        self.im_filenames = {
            2018: ['merged_2018_crop.tif'],
            2019: ['S2_PI_B1_B2_2019_crop.tif',
                   'S2_PI_B3_B4_2019_crop.tif',
                   'S2_PI_B5_B6_2019_crop.tif',
                   'S2_PI_B7_B8_2019_crop.tif',
                   'S2_PI_B8A_B9_2019_crop.tif',
                   'S2_PI_B10_B11_2019_crop.tif',
                   'S2_PI_B12_2019_crop.tif'],
            2020: ['merged_2020_crop.tif']
        }
        '''
        self.im_filenames = {
            2017: ['S2_PI_B4_B3_2017_crop.tif',
                   'S2_PI_B2_B8_2017_crop.tif'],
            2018: ['merged_2018_crop.tif'],
            2019: ['S2_PI_B4_B3_2019_crop.tif',
                   'S2_PI_B2_B8_2019_crop.tif'],
            2020: ['S2_PI_B4_B3_2020_crop.tif',
                   'S2_PI_B2_B8_2020_crop.tif'],
            2021: ['S2_PI_B4_B3_2021_crop.tif',
                   'S2_PI_B2_B8_2021_crop.tif']                   
        }
        self.biome_limits = os.path.join(self.optical_im_folder, 'biome_limits.tif')

        landsat_base = 'D:/Jorge/datasets/landsat/PI/'
        self.landsat = [
            os.path.join(landsat_base, 'L8_PI_2019.tif'),
            os.path.join(landsat_base, 'L8_PI_2020.tif'),
            os.path.join(landsat_base, 'L8_PI_2021.tif'),

        ]
class PathsL8MT(Paths):
    def __init__(self): 
        self.reference_folder = 'D:/Jorge/datasets/deforestation/L8MT/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/L8MT/'

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/L8MT/' 

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2008_L8MT.tif') 

        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')
 
        self.deforestation_time = {
            2022: os.path.join(self.reference_folder, 'deforestation_time_normalized_2022.npy'),
            2021: os.path.join(self.reference_folder, 'deforestation_time_normalized_2021.npy'),

        } 

        self.optical_im_past_dates = { 
            2023: os.path.join(self.optical_im_folder,  '2023'), 
            2022: os.path.join(self.optical_im_folder,  '2022'),
            2021: os.path.join(self.optical_im_folder,  '2021'),
        }

        self.cloud_mask = {
            2023: os.path.join(self.optical_im_folder,  '2023', 'cloudmask_L8MT_2023.npy'), 
            2022: os.path.join(self.optical_im_folder,  '2022', 'cloudmask_L8MT_2022.npy'),
            2021: os.path.join(self.optical_im_folder,  '2021', 'cloudmask_L8MT_2021.npy'),
        }

        self.im_filenames = {
            2021: ['LC08_L1TP_228067_20210720_20210729_02_T1_B1_crop.TIF',
                   'LC08_L1TP_228067_20210720_20210729_02_T1_B2_crop.TIF',
                   'LC08_L1TP_228067_20210720_20210729_02_T1_B3_crop.TIF',
                   'LC08_L1TP_228067_20210720_20210729_02_T1_B4_crop.TIF',
                   'LC08_L1TP_228067_20210720_20210729_02_T1_B5_crop.TIF',
                   'LC08_L1TP_228067_20210720_20210729_02_T1_B6_crop.TIF',
                   'LC08_L1TP_228067_20210720_20210729_02_T1_B7_crop.TIF'],
            2022: ['LC09_L1TP_228067_20220731_20230405_02_T1_B1.TIF',
                   'LC09_L1TP_228067_20220731_20230405_02_T1_B2.TIF',
                   'LC09_L1TP_228067_20220731_20230405_02_T1_B3.TIF',
                   'LC09_L1TP_228067_20220731_20230405_02_T1_B4.TIF',
                   'LC09_L1TP_228067_20220731_20230405_02_T1_B5.TIF',
                   'LC09_L1TP_228067_20220731_20230405_02_T1_B6.TIF',
                   'LC09_L1TP_228067_20220731_20230405_02_T1_B7.TIF'],
            2023: ['LC09_L1TP_228067_20230803_20230804_02_T1_B1_crop.TIF',
                'LC09_L1TP_228067_20230803_20230804_02_T1_B2_crop.TIF',
                'LC09_L1TP_228067_20230803_20230804_02_T1_B3_crop.TIF',
                'LC09_L1TP_228067_20230803_20230804_02_T1_B4_crop.TIF',
                'LC09_L1TP_228067_20230803_20230804_02_T1_B5_crop.TIF',
                'LC09_L1TP_228067_20230803_20230804_02_T1_B6_crop.TIF',
                'LC09_L1TP_228067_20230803_20230804_02_T1_B7_crop.TIF']
        }

class PathsL8AM(Paths):
    def __init__(self): 
        self.reference_folder = 'D:/Jorge/datasets/deforestation/L8AM/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/L8AM/'

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/L8AM/' 

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2008_L8AM.tif') 

        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')
 
        self.deforestation_time = {
            2022: os.path.join(self.reference_folder, 'deforestation_time_normalized_2022.npy'),
            2021: os.path.join(self.reference_folder, 'deforestation_time_normalized_2021.npy'),

        } 

        self.optical_im_past_dates = { 
            2023: os.path.join(self.optical_im_folder,  '2023'), 
            2022: os.path.join(self.optical_im_folder,  '2022'),
            2021: os.path.join(self.optical_im_folder,  '2021'),
        }

        self.cloud_mask = {
            2023: os.path.join(self.optical_im_folder,  '2023', 'cloudmask_L8AM_2023.npy'), 
            2022: os.path.join(self.optical_im_folder,  '2022', 'cloudmask_L8AM_2022.npy'),
            2021: os.path.join(self.optical_im_folder,  '2021', 'cloudmask_L8AM_2021.npy'),
        }

        self.im_filenames = {
            2021: ['LC08_L1TP_232066_20210817_20210827_02_T1_B1_crop.TIF',
                   'LC08_L1TP_232066_20210817_20210827_02_T1_B2_crop.TIF',
                   'LC08_L1TP_232066_20210817_20210827_02_T1_B3_crop.TIF',
                   'LC08_L1TP_232066_20210817_20210827_02_T1_B4_crop.TIF',
                   'LC08_L1TP_232066_20210817_20210827_02_T1_B5_crop.TIF',
                   'LC08_L1TP_232066_20210817_20210827_02_T1_B6_crop.TIF',
                   'LC08_L1TP_232066_20210817_20210827_02_T1_B7_crop.TIF'],
            2022: ['LC09_L1TP_232066_20220727_20230406_02_T1_B1.TIF',
                   'LC09_L1TP_232066_20220727_20230406_02_T1_B2.TIF',
                   'LC09_L1TP_232066_20220727_20230406_02_T1_B3.TIF',
                   'LC09_L1TP_232066_20220727_20230406_02_T1_B4.TIF',
                   'LC09_L1TP_232066_20220727_20230406_02_T1_B5.TIF',
                   'LC09_L1TP_232066_20220727_20230406_02_T1_B6.TIF',
                   'LC09_L1TP_232066_20220727_20230406_02_T1_B7.TIF'],
            2023: ['LC08_L1TP_232066_20230722_20230803_02_T1_B1_crop.TIF',
                'LC08_L1TP_232066_20230722_20230803_02_T1_B2_crop.TIF',
                'LC08_L1TP_232066_20230722_20230803_02_T1_B3_crop.TIF',
                'LC08_L1TP_232066_20230722_20230803_02_T1_B4_crop.TIF',
                'LC08_L1TP_232066_20230722_20230803_02_T1_B5_crop.TIF',
                'LC08_L1TP_232066_20230722_20230803_02_T1_B6_crop.TIF',
                'LC08_L1TP_232066_20230722_20230803_02_T1_B7_crop.TIF']
        }


class PathsMO(Paths):
    def __init__(self): 

        self.reference_folder = 'D:/Jorge/datasets/deforestation/MO/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/MO/'

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/MO/' 

        self.deforestation_before_2008 = os.path.join(self.reference_folder, 'deforestation_before_2000_MO.tif') 

        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')
 
        self.deforestation_time = {
            2020: os.path.join(self.reference_folder, 'deforestation_time_normalized_2020.npy'),
            2019: os.path.join(self.reference_folder, 'deforestation_time_normalized_2019.npy'),
            2018: os.path.join(self.reference_folder, 'deforestation_time_normalized_2018.npy'),
        } 

        self.optical_im_past_dates = { 
            2020: os.path.join(self.optical_im_folder,  '2020'), 
            2019: os.path.join(self.optical_im_folder,  '2019'),
            2018: os.path.join(self.optical_im_folder,  '2018'),
        }

        self.cloud_mask = {
            2020: os.path.join(self.optical_im_folder,  '2020', 'cloudmask_MO_2020.npy'), 
            2019: os.path.join(self.optical_im_folder,  '2019', 'cloudmask_MO_2019.npy'),
            2018: os.path.join(self.optical_im_folder,  '2018', 'cloudmask_MO_2018.npy'),
        }

        self.im_filenames = {
            2018: ['S2_MO_B4_B3_2018_crop.tif',
                   'S2_MO_B2_B8_2018_crop.tif'],
            2019: ['S2_MO_B4_B3_2019_crop.tif',
                   'S2_MO_B2_B8_2019_crop.tif'],
            2020: ['S2_MO_B4_B3_2020_crop.tif',
                   'S2_MO_B2_B8_2020_crop.tif']
        }
        self.biome_limits = os.path.join(self.optical_im_folder, 'biome_limits.tif')

class PathsMA(Paths): 
    def __init__(self): 
        self.reference_folder = 'D:/Jorge/datasets/deforestation/MA/'
        self.optical_im_folder = 'D:/Jorge/datasets/sentinel2/MA/'

        self.experiment = 'D:/Jorge/datasets/deforestation/experiments/MA/' 

        # deforestation_before_2008 already in reference
        self.hydrography = os.path.join(self.reference_folder, 'hydgrography.tif')
        self.deforestation_past_years = os.path.join(self.reference_folder, 'deforestation_past_years.tif')
 
        self.deforestation_time = {
            2021: os.path.join(self.reference_folder, 'deforestation_time_normalized_2021.npy'),
            2020: os.path.join(self.reference_folder, 'deforestation_time_normalized_2020.npy'),
        } 

        self.optical_im_past_dates = { 
            2021: os.path.join(self.optical_im_folder,  '2021'), 
            2020: os.path.join(self.optical_im_folder,  '2020'), 
        }

        self.cloud_mask = {
            2021: os.path.join(self.optical_im_folder,  '2021', 'cloudmask_MA_2021.npy'), 
            2020: os.path.join(self.optical_im_folder,  '2020', 'cloudmask_MA_2020.npy'), 
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