# QMR-FS
Select linearly independent features using the QMR decomposition.


## Install

First install dependencies, either the pip `requirements.txt` file, or the conda `environment.yml` file.
The file `requirements.txt` is minimalistic, containing only the packages needed to run QMR-FS and the experiments.
The conda file also contains plotting libraries, jupyterlab, and some libraries to make Numba faster.

Then install this package in your environment:
```commandline
pip install -e .
```


## Run QMR-FS
QMR-FS can be run simply as 
```python
import numpy as np
import torch
import qmrfs.qmr_feature_selection as qmrfs
features = np.random.rand(20, 10)
selected_features, _, _ = qmrfs.qmr_fs(
            features=features,
            tolerance=1e-1,
            sorting_strategy='entropy_high2low',
            device=torch.device('cpu')
        )
```

## Datasets
The small UCI ML datasets will be downloaded automatically. 
The large datasets will have to be downloaded manually and extracted in to the `datasets` folder.
Isolet is available as part of the UFS Matlab toolbox (see below). 
It needs to be put into the folder `datasets/isolet`.

## Run experiments
Save all the datasets to disk
```commandline
python scripts/run_feature_selection.py --save-data
```

Save the QMR-FS features to disk
```commandline
python scripts/run_feature_selection.py --qmrfs
```

Run the classification and clustering evaluation for each of the saved features
```commandline
python scripts/main_experiment.py
```

The script `main_experiment.py` will run the evaluation for the QMR-FS features, and any other features saved into the `baseline_features` folder as `.mat` files.
See below for instructions to save features of other UFS methods.

### Run other UFS methods: SVD-entropy and USFSM
SVD-entropy and USFSM have been implemented in python as part of this repo. Run them for all datasets:
```commandline
python scripts/run_feature_selection.py --baselines
```
Note that USFSM can be very slow.


### Run other UFS methods: Matlab
Make sure you have Matlab installed, and download the UFS toolbox: https://se.mathworks.com/matlabcentral/fileexchange/116370-unsupervised-feature-selection-toolbox.

Unpack the toolbox folders and files into the `ufstoolbox` folder, next to the `models` folder present there.
Add the folder `usftoolbox` to the Matlab path.
Make sure that you have run the function `save_datafiles_as_mat()` in `scripts/run_feature_selection.py` first.
Then you can run the Matlab UFS methods in Matlab with the scripts `run_matlab_baselines.m` and `run_matlab_baselines_isolet.m`.
This saves the selected features for each dataset and method into the `baseline_features` folder.
