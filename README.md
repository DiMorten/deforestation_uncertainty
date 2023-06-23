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

## Folder structure



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


## Inference on Ensemble:

1. Set the data in the folder structure from section _Folder structure_
2. Configure the training run using the _config_ dictionary. 
    - training: If True, training and inference is done. If False, only inference
    - inference_times: Number of inference times for the ensemble
    - uncertainty_method: Select uncertainty metric. Options: "pred_entropy": Predictive entropy, "pred_var": Predictive variance, "MI": Mutual Information, "KL": Kullback-Leibler Divergence. Default: "pred_entropy"
    - removePolygons: If True, remove polygons with an area smaller to 6.25ha, following PRODES methodology. Default: True
    - 

    - 
4. Run train_ensemble.ipynb
