# Getting started

## Workshop notebooks in colab

### Open the notebook

First notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VisiumCH/AMLD-2021-Sketchy/blob/master/notebooks/workshop/AMLD-2021-Sketchy-Demo1_Training.ipynb)

Second notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VisiumCH/AMLD-2021-Sketchy/blob/master/notebooks/workshop/AMLD-2021-Sketchy-Demo2_Performance.ipynb)

## Run the codebase locally

Clone the repo in your working machine and set up the environment.

```bash
git clone git@github.com:VisiumCH/AMLD-2021-Sketchy.git AMLD-2021-Sketchy
cd AMLD-2021-Sketchy
make env
source env/bin/activate
make init
```

From there, you should have a folder named **io/** at the root of the project. This folder is a symlink to **/io/** so you can share the raw (and processd) data as well as the resulting models.
You should have the data in 'io/models/quickdraw_models/' folder.

### Run the Web App locally

To install the web application (the first time only),

```bash
cd web_app/sketchy
npm install sketchy
```

To start using the application,

1. Start the server:

```bash
python src/api/server.py
```

You should expect to wait a moment (around 5 minutes) for the server to be ready after this command.
The web app might be irresponsive for a moment (finishing to prepare) if you open it too quickly.

3. Start the web application: open a new terminal (without stopping the server!) and:

```bash
cd web_app/sketchy
npm start
```

### Training and inference of a new model

Training a model takes a lot of time (quickdraw_training, used in the workshop, took almost a months to train for instance). So it is not done in the workshop notebooks.

All explanations therefore are in [src/README.md](https://github.com/VisiumCH/AMLD-2021-Sketchy/blob/workshop_notebook/src/README.md) of this repository.

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
    │   │   │   │   └── theme.js                <- define the default colors if the web app
    │   │   │   └── interactionPages
    │   │   │   │   └── drawing.js              <- /drawing page: user draw sketch and images are retrieved
    │   │   │   │   └── embeddingsPlot.js       <- /embeddings page: play around with different embedding projections
    │   │   │   │   └── seeDataset.js           <- / page: see the different datasets data and classes
    │   │   │   └── performancePages
    │   │   │   │   └── scalarPerformance.js    <- /scalar_perf page: see the scalar performance of the model (plots)
    │   │   │   │   └── imagePerformance.js     <- /image_perf page: see the inference and attention at different epochs
    │   │   │   └── App.js                      <- define the route of the pages
    │   │   │   └── constants.js                <- constants of web app
    │   │   │   └── drawer.js                   <- drawer to choose page to go to
    │   │   │   └── index.js                    <- root of the app
    │   │   │   └── ui_utils.js                 <- define ui that are used in multiple pages
