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

---

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
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── api                 <- Scripts to serve model predictions via API
    |   |   └── test_scaling         <- tests with locust to assess the number of people that can use the app at the same time
    │   │   └── api_dim_reduction.py <- class to return the projected embeddings and clicked images
    │   │   └── api_inference.py     <- class for returning closest images and attention to drawn sketch
    │   │   └── api_options.py       <- command line options to load chosen model for training
    │   │   └── api_performance.py   <- class for returning the model performance (graphs and images at all epochs)
    │   │   └── server.py            <- implements all the flask api to communicate with the web app
    │   │   └── utils.py             <- utils function for the web app server
    │   |
    │   ├── data               <- Scripts to prepare and load datasets
    │   │   └── composite_dataset.py <- Class to load multiple dataset in the same training (Sketchy + TU-Berlin) or (Sketchy + TU-Berlin + Quickdraw)
    │   │   └── default_dataset.py <- Default class to load a dataset (Sketchy, TU-Berlin, Quickdraw)
    │   │   └── loader_factory.py  <- Factory to load chosen dataset
    │   │   └── utils.py           <- utils function for data
    |   |
    │   ├── models            <- Scripts to train and test models and then use trained models to make predictions
    │   │   ├── inference
    │   │   │   └── inference.py             <- inference script to ouptput visual results
    │   │   │   └── preprocess_embeddings.py <- preprocess embeddings in advance for inference
    │   │   └── encoder.py    <- structure of the encoder network
    │   │   └── logs.py       <- logger for tensorboard: scalar metrics, attention plot, embeddings projector
    │   │   └── loss.py       <- loss of the network
    │   │   └── metrics.py    <- metrics (mean average precision, mean average precision, precision)
    │   │   └── test.py       <- test to log metrics of a model
    │   │   └── train.py      <- training: starts and handles the training
    │   │   └── utils.py      <- utils function for data
    │   │
    │   └── constants.py   <- Contains all constants of the projects
    │   └── options.py     <- Contains all meta information that can be passsed as command line arguments
    │
    ├── web_app            <- React web application folder
    │   ├── sketchy        <- Scripts to serve model predictions via API
    |   |   └── public     <- index and icons
    |   |   └── src        <- Source code for the web app
    │   │   │   └── styles
    │   │   │   │   └── components
    │   │   │   │   │   │   └── buttonStyles.js <- define the style of the buttons
    │   │   │   │   │   └── theme.js            <- define the default colors if the web app
    │   │   │   └──
