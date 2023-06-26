# This is the code for the paper "Semi-Automatic Monitoring of Deforestation in the Brazilian Amazon: Uncertainty Estimation and Characterization of High Uncertainty Areas"

## Installation

These instructions were tested in Windows 10

1. We provide the `environment.yml` file. In conda, install the environment running: `conda env create -f environment.yml`
2. Activate the environment: `conda activate tf2`
3. Install GDAL 3.4.3 for Python 3.9
   - Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
   - Download `GDAL-3.4.3-cp39-cp39-win_amd64.whl`
   - Install running: `pip install GDAL-3.4.3-cp39-cp39-win_amd64.whl`
4. Install rasterio 1.2.10 for Python 3.9
    - Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
    - Download `rasterio-1.2.10-cp39-cp39-win_amd64.whl`
    - Install running: `pip install rasterio-1.2.10-cp39-cp39-win_amd64.whl`
5. Run `pip install -U numpy`

## Dataset folder structure
```
.
├── datasets/                                                                       # Dataset folder
│   ├── deforestation/                                                              # Deforestation detection reference
│   │   ├── deforestation_before_2008/deforestation_before_2008_para.tif            # Reference with deforestation before 2008
│   │   ├── PA/deforestation_past_years.tif                                         # Reference with deforestation from 2008 until latest or present year
│   ├── sentinel2/                                                                  # Input rasters for the network      
│   │   ├── PA_2017/                                                                # Data for site PA and date 2017
│   │   │   ├──cloudmask_PA_2017.npy                                                # Cloud mask (See cloud mask section)
│   │   │   ├──PA_S2_2017_B1_B2_B3_crop.tif                                         # bands 1,2,3 for Para 2017
│   │   │   ├──PA_S2_2017_B4_B5_B6_crop.tif                                         # bands 4,5,6 for Para 2017
│   │   │   └── ...
│   │   ├── PA_2018/          
│   │   │   ├──cloudmask_PA_2018.npy
│   │   │   ├──COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif                          # bands 1,2,3 for Para 2018
│   │   │   ├──COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif                          # bands 4,5,6 for Para 2018
│   │   │   └── ...
│   │   ├── PA_2019/   
│   │   │   ├──cloudmask_PA_2019.npy
│   │   │   ├──COPERNICUS_S2_20190721_20190726_B1_B2_B3.tif                          # bands 1,2,3 ofPara 2019
│   │   │   ├──COPERNICUS_S2_20190721_20190726_B4_B5_B6.tif                          # bands 1,2,3 ofPara 2019
│   │   │   └── ...
│   ├── landsat/                                                                     # Only for visualization as in Figures 7 to 10      
│   │   ├── PA/                                                                      # PA site
│   │   │   ├── landsat_PA_2018.tif                                                  # Landsat T_{-1} image
│   │   │   ├── landsat_PA_2019.tif                                                  # Landsat T_0 image
│   │   │   └── landsat_PA_2020.tif                                                  # Landsat T_1 image
│   └── 
└── ...
```
## Calculate cloud mask
Edit the `config` dictionary for configuration.
   - "dataset": Options: "PA": Para site. "MT": Mato Grosso site.
   - "year": year from the cloud mask to be calculated. Example: 2019
     
The script will generate a NPY with the cloud mask which is used during training.

Example: to train in PA site for [2018, 2019] dates, execute the script two times, once for each year (2018 and 2019)

## Calculate temporal distance to past deforestation 

"Temporal distance to past deforestation" is used as an input to the network as explained in Section 2.3.
To calculate it, use the script `preprocess_deforestation_time.py`

Edit the `config` dictionary for configuration.
   - "dataset": Options: "PA": Para site. "MT": Mato Grosso site.
   - "year": year from the latest image in the image pair. Example: 2019
     
The script will generate a NPY with the cloud mask which is used during training.

Example: to train in PA site for [2018, 2019] dates, execute the script only for the latest year (2019)

## Generate the normalized input data

Normalized input is pre-calculated. Use `preprocess_dataset.py` to normalize the input image.

Edit the `config` dictionary for configuration.
   - "dataset": Options: "PA": Para site. "MT": Mato Grosso site.
   - "year": year from the latest image in the image pair. Example: 2019
     
The script will generate a NPY with the normalized input image which is used during training.

Example: to train in PA site for [2018, 2019] dates, execute the script two times, once for each year (2018 and 2019)

## MCD (Execute a single experiment)

1. Set the data in the folder structure from section _Folder structure_
2. Open the train_mc_dropout.ipynb notebook
3. Configure the training run using the _config_ dictionary. 
    - training: If True, training and inference is done. If False, only inference
    - inference_times: Number of inference times for the MC Dropout calculation
    - uncertainty_method: Select uncertainty metric. Options are presented next. Default: "pred_entropy" 
        - "pred_entropy": Predictive entropy, 
        - "pred_var": Predictive variance, 
        - "MI": Mutual Information, 
        - "KL": Kullback-Leibler Divergence. 
    - removePolygons: If True, remove polygons with an area smaller to 6.25ha, following PRODES methodology. Default: True
    - site: Dataset site. Options are presented next. Default: "PA"
        - "MT" (Mato Grosso)
        - "PA" (Para)
    - training_date: Training date pair. Options: "earlier" for training with an earlier pair of dates, and "current" for training and testing on the same pair of dates. Default: "earlier"
4. Run all

-------------
## Run in batch (Multiple executions)
This script allows training for single run, MCD and ensemble methods.
It also allows inference for single run and MCD methods.
For inference on ensemble method go to the next section.

In the paper, multiple training and inference runs are applied for each uncertainty method (10 repetitions). To run those repetitions, use `train_grid_execution.ipynb`

1. Set the data in the folder structure from section _Folder structure_
2. Open the `train_grid_execution.ipynb` notebook
3. Configure the training run using the _config_ dictionary. 
    - `training`: If True, training is done. Default: False
    - `inferring`: If True, inference is done. Default: True
    - `site`: Dataset site. Options are presented next. Default: "PA"
        - "MT" (Mato Grosso)
        - "PA" (Para)
    - `training_date`: Training with current or earlier pair of dates. Options: "current", "earlier". Default: "earlier"
    - `mode`: Uncertainty mode. Options are presented next. Default: 'mcd' (Monte Carlo Dropout)
        - mcd: Monte Carlo Dropout
        - single_run: Entropy from a single inference run
        - ensemble: Ensemble (Only training. For inference, use train_ensemble.ipynb)
   - inference_times: Number of inference times for the ensemble
   - uncertainty_method: Select uncertainty metric. Options: "pred_entropy": Predictive entropy, "pred_var": Predictive variance, "MI": Mutual Information, "KL": Kullback-Leibler Divergence. Default: "pred_entropy"


## Inference on Ensemble:

1. Set the data in the folder structure from section _Folder structure_
2. Configure the training run using the _config_ dictionary. 
    - training: If True, training and inference is done. If False, only inference
    - inference_times: Number of inference times for the ensemble
    - uncertainty_method: Select uncertainty metric. Options: "pred_entropy": Predictive entropy, "pred_var": Predictive variance, "MI": Mutual Information, "KL": Kullback-Leibler Divergence. Default: "pred_entropy"
    - removePolygons: If True, remove polygons with an area smaller to 6.25ha, following PRODES methodology. Default: True

4. Run train_ensemble.ipynb
