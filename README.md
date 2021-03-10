# Getting started
Clone the repo in your working machine (usually Google Cloud instance).


```bash
make env
source env/bin/activate
make init
```
From there, you should have a folder named **io/** at the root of the project. This folder is a symlink to **/io/** so you can share the raw (and processd) data as well as the resulting models.


If the Google bucket has not been pulled already, or data in bucket has been updated recently:

```bash
make sync_raw_data
```


# Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── io                 <- All input/output files, it is a symlink to the attached disk /io
    |   ├── data
    |   │   ├── external   <- Data from third party sources.
    |   │   ├── interim    <- Intermediate data that has been transformed.
    |   │   ├── processed  <- The final, canonical data sets for modeling.
    |   │   └── raw        <- The original, immutable data dump.
    |   |   |   └── Sketchy        <- Sketchy dataset
    |   |   |   │   └── images     <- contains all sketchy images
    |   |   |   │   └── sketches   <- contains all sketchy sketches
    |   |   |   └── TU-Berlin      <- TU-Berlin dataset
    |   |   |   │   └── images       
    |   |   |   │   └── sketches 
    |   |   |   └── Quickdraw      <- TU-Berlin dataset
    |   |   |   │   └── images       
    |   |   |   │   └── sketches 
    |   ├── models         <- Trained and serialized models, model predictions, or model summaries
    |   |   |   └── X_model
    |   |   |   │   └── preprocess_embeddings             <- folder with preprocessed embeddings based on this model
    |   |   |   |   |   └── embeddings*dict_class.json    <- Dictionnary of * dataset 
    |   |   |   |   |   └── embeddings*array.npy          <- Array of pre-computed embeddings of * dataset
    |   |   |   |   |   └── embeddings*meta.csv           <- Meta information (file_path and class number) of * dataset
    |   |   |   │   └── prediction                       <- Folder with predictions plot using the model
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. 
    │    └── 1.0-pml-AttentionPlot.ipynb     <- Notebook to observe/modify with attention plots
    │    └── 1.0-pml-Inference.ipynb         <- Notebook for inference of a model.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api            <- Scripts to serve model predictions via API
    │   │   └── server.py
    │   |
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │   └── default_dataset.py <- Default class to load a dataset (Sketchy, tU-Berlin, Quickdraw)
    │   │   └── loader_factory.py  <- Factory to load chosen dataset
    │   │   └── sktu_extended.py   <- Class to load both Sketchy and TU-Berlin in the same training
    │   │   └── sktuqg_extended.py <- Class to load Sketchy, TU-Berlin and Quickdraw in the same training
    │   │   └── utils.py           <- utils function for data
    │   │
    │   ├── tests         <- Tests for src content
    │   │   ├── test_all.py
    │   │   └── test_api.py
    │   │   └── test_data.py
    │   │   └── test_models.py
    │   │   └── test_visualization.py
    |   |
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── inference
    │   │   │   └── inference.py             <- inference script to ouptput visual results
    │   │   │   └── preprocess_embeddings.py <- preprocess embeddings in advance for inference
    │   │   └── encoder.py    <- structure of the encoder network
    │   │   └── logs.py       <- logger for tensorboard: scalar metrics, attention plot, embeddings projector
    │   │   └── loss.py       <- loss of the network
    │   │   └── metrics.py    <- metrics (mean average precision, mean average precision@200, precision@200)
    │   │   └── test.py       <- test to log metrics of a model
    │   │   └── train.py      <- training: starts and handles the training
    │   │   └── utils.py      <- utils function for data
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │   └── options.py  <- Contains all constants and meta information
