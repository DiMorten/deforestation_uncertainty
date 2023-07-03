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
5. Run `pip3 install numpy==1.23.5`

## Dataset folder structure
```
.
├── datasets/                                                                  # Dataset folder
│   ├── deforestation/                                                         # Deforestation detection reference
│   │   ├── PA/                                                                # Reference for PA site
│   │   │   ├── deforestation_before_2008_PA.tif                               # Reference with deforestation before 2008
│   │   │   ├── deforestation_past_years.tif                                   # Reference with deforestation from 2008 until latest or present year
│   │   │   ├── deforestation_time_normalized_2018.npy                         # See section "Calculate temporal distance to past deforestation"
│   │   └── └── deforestation_time_normalized_2019.npy                         # See section "Calculate temporal distance to past deforestation"
│   ├── sentinel2/                                                             # Input rasters for the network      
│   │   ├── PA/                                                                # Sentinel2 data for PA site
│   │   │   ├── 2017/                                                          # Data for PA site and 2017 date (T_{-1})
│   │   │   │   ├──cloudmask_PA_2017.npy                                       # Cloud mask (See cloud mask section)
│   │   │   │   ├──PA_S2_2017_B1_B2_B3_crop.tif                                # bands 1,2,3 for PA 2017 (T_{-1})
│   │   │   │   ├──PA_S2_2017_B4_B5_B6_crop.tif                                # bands 4,5,6 for PA 2017 (T_{-1})
│   │   │   │   └── ...
│   │   │   ├── 2018/          
│   │   │   │   ├──cloudmask_PA_2018.npy
│   │   │   │   ├──COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif                # bands 1,2,3 for PA 2018 (T_0)
│   │   │   │   ├──COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif                # bands 4,5,6 for PA 2018 (T_0)
│   │   │   │   └── ...
│   │   │   ├── 2019/   
│   │   │   │   ├──cloudmask_PA_2019.npy
│   │   │   │   ├──COPERNICUS_S2_20190721_20190726_B1_B2_B3.tif                # bands 1,2,3 for PA 2019 (T_1)
│   │   │   │   ├──COPERNICUS_S2_20190721_20190726_B4_B5_B6.tif                # bands 4,5,6 for PA 2019 (T_1)
│   │   └── └── └── ...
│   ├── landsat/                                                               # Only for visualization as in Figures 7 to 10      
│   │   ├── PA/                                                                # PA site
│   │   │   ├── landsat_PA_2018.tif                                            # Landsat T_{-1} image
│   │   │   ├── landsat_PA_2019.tif                                            # Landsat T_0 image
│   └── └── └── landsat_PA_2020.tif                                            # Landsat T_1 image
└── 
```

## Dataset downloading

- Download the ROI `.shp` shapefile for PA and MT sites at https://drive.google.com/drive/folders/1O8Ivuc2JVB-iSMVrA5TXJRK2BZfOqxO7?usp=sharing

1. Input image downloading

- Download each Sentinel-2 image using the link: 
https://code.earthengine.google.com/2c31e8b0000a34fbc70c9e8d80dd7237
- Upload the image ROI (Region of interest) as a `.shp` file, and load it in the `polygon` variable. 
`S2_collection` is a list of images within the specified date range. Select the date range specified in Table 1. 
The function `exportBands` exports the selected bands to a TIF file. To reduce computational complexity, export 3 bands per TIF file (Create separate TIF for bands 1,2,3, another for 4,5,6, another for 7,8,9, and another for 10,11,12).

  
2. Obtaining reference

- Go to http://terrabrasilis.dpi.inpe.br/downloads/
- Yearly deforestation between 2008 and 2022: Download "Incremento anual no desmatamento - Shapefile (2008/2022)" as a SHP file. Convert it to raster and crop its dimensions to the desired ROI using QGIS. Save the TIF file in `dataset/deforestation/{site}/deforestation_past_years.tif`
- Deforestation before 2008: Download "Máscara de área acumulada de supressão da vegetação nativa - Shapefile (2007)" as a SHP file. Convert it to raster and crop its dimensions to the desired ROI using QGIS. Save the TIF file in `dataset/deforestation/{site}/deforestation_before_2008_{site}.tif`

## Calculate cloud mask

Use the `cloud_mask.py` script.
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

## Run in batch (Multiple executions)
This script allows training for single run, MCD and ensemble methods.
It also allows inference for single run and MCD methods.
For inference on ensemble method go to the next section.

In the paper, multiple training and inference runs are applied for each uncertainty method (10 repetitions). To run those repetitions, use `train_grid_execution.ipynb`

1. Set the data in the folder structure from section _Folder structure_
2. Open the `train_grid_execution.ipynb` notebook
3. Configure the training run using the _config_ dictionary. 
    - `training`: If True, training is done. Default: False
    - `training_times`: If training, specify number of training runs
    - `inferring`: If True, inference is done. Default: True
    - `site`: Dataset site. Options are presented next. Default: "PA"
        - "MT" (Mato Grosso)
        - "PA" (Para)
    - `training_date`: Training with current or earlier pair of dates. Options: "current", "earlier". Default: "earlier"
    - `mode`: Uncertainty mode. Options are presented next. Default: 'mcd' (Monte Carlo Dropout)
        - mcd: Monte Carlo Dropout
        - single_run: Entropy from a single inference run
        - ensemble: Ensemble (Only training. For inference, use train_ensemble.ipynb)
   - `inference_times`: Number of inference times for the ensemble
   - `uncertainty_method`: Select uncertainty metric. Options: "pred_entropy": Predictive entropy, "pred_var": Predictive variance, "MI": Mutual Information, "KL": Kullback-Leibler Divergence. Default: "pred_entropy"

### Analyzing log results

The batch running script `train_grid_execution.ipynb` produces a log with metrics for each experiment repetition. Log is located in `output/log/log_{method}.pkl`. To observe its resulting metrics, use `log_analyze.ipynb`. Specify the log to analyze in the `filenames` variable as: `filenames = ['log_{method}.pkl]`. Results will be saved to CSV in `output/log/results.csv`.

## Visualizing a single MCD, ensemble or single run experiment:

Use the script `run_single_experiment.ipynb`

1. Set the data in the folder structure from section _Folder structure_
2. Open the `run_single_experiment.ipynb` notebook
3. Configure the training run using the _config_ dictionary. 
    - `training`: If True, training is done.
    - `inferring`: If True, inference is done.
    - `site`: Dataset site. Options are presented next. Default: "PA"
        - "MT" (Mato Grosso)
        - "PA" (Para)
    - `training_date`: Training date pair. Options: "earlier" for training with an earlier pair of dates, and "current" for training and testing on the same pair of dates.
    - `mode`: Uncertainty mode. Values: "mcd" (Monte Carlo Dropout), "ensemble", "single_run" (Baseline). Default: "ensemble"
    - `inference_times`: Number of inference times is automatically set to 10 for "mcd" and "ensemble", and 1 for "single_run"
    - `uncertainty_method`: Select uncertainty metric. Options are presented next. Default: "pred_entropy" 
        - "pred_entropy": Predictive entropy, 
        - "pred_var": Predictive variance, 
        - "MI": Mutual Information, 
        - "KL": Kullback-Leibler Divergence.
        - "pred_entropy_single": Used for "single run" experiment (Baseline). Automatically set if 
    - `removePolygons`: If True, remove polygons with an area smaller to 6.25ha, following PRODES methodology. Default: True
 Default: "earlier"
4. Run all

-------------

