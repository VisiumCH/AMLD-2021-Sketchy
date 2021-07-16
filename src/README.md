# Documentation per folder

First, always make sure that the file `options.py` contains the right metadata and constants for the task, you can modify the arguments directly from the command line.

##### Replacing \*training-name\* by the name you want to give to the training.

The model will be saved in io/models/\*training-name\*

## data/

### loader_factory.py

To check that all dataset are loading well, you can call

```bash
python src/data/loader_factory.py *any-training-name*
```

It will load every dataset and print their length. Also, it creates an example image of the positive and negative images created for training. It can be seen in the /io/data/processed/ folder.

The datasets are:

- sketchy: for sketchy dataset
- tuberlin: for TU-Berlin dataset
- quickdraw: for Quickdraw dataset
- sk+tu: for combining Sketchy and TU-Berlin datasets
- sk+tu+qd: for combining Sketchy, TU-Berlin and Quickdraw datasets

You should expect to see:

Dataset sketchy:

        * Length sketch train: 60335
        * Length image train: 58396
        * Length sketch valid: 7543
        * Length image valid: 7298
        * Length sketch test: 7603
        * Length image test: 7308

Dataset tuberlin

        * Length sketch train: 16000
        * Length image train: 163159
        * Length sketch valid: 2000
        * Length image valid: 20393
        * Length sketch test: 2000
        * Length image test: 20518

Dataset sk+tu

        * Length sketch train: 76335
        * Length image train: 221555
        * Length sketch valid: 9543
        * Length image valid: 27691
        * Length sketch test: 9603
        * Length image test: 27826

Dataset quickdraw

        * Length sketch train: 265809
        * Length image train: 162367
        * Length sketch valid: 33226
        * Length image valid: 20293
        * Length sketch test: 33336
        * Length image test: 20351

Dataset sk+tu+qd

        * Length sketch train: 342144
        * Length image train: 383922
        * Length sketch valid: 42769
        * Length image valid: 47984
        * Length sketch test: 42939
        * Length image test: 48177

---

## models/

### train

To train a model on a dataset, verify options.py for your settings, then

```bash
python src/models/train.py *training-name* *any additional argument*
```

the model and logs will be saved in a folder called name in io/models/name/.

### test

To test a dataset on a model,

```bash
python src/models/test.py *training-name*
```

### inference

The inference needs two steps.

1. First, the embeddings of all images must be pre-computed and save with their metadata. Therefore:

```bash
python src/models/inference/preprocess_embeddings.py *training-name*
```

the embeddings will be saved in the same folder as the model folder in a folder called precomputed_embeddings.

2. Then,

```bash
python src/models/inference/inference.py *training-name*
```

selects 20 random sketches. For each sketch, computes the embedding and the 4 closest embeddings of images. It then plots the sketches and the associated close images and saves them in the same folder as the model folder in a folder called predictions. (/io/models/_training_name_/predictions)

---

## api/

The web application calls the api in server.py.
if it is the first time, you will need to install all the react libraries

```bash
npm install sketchy
```

To run the web app, you must have already trained a model and precomputed its embeddings:

```bash
python src/models/train.py *training-name*
python src/models/inference/preprocess_embeddings.py *training-name*
```

then, do the following

### server.py

```bash
python src/api/server.py *training-name*
```

You should expect to wait a moment (< 5 minutes) for the server to be ready after this command.
The web app might be irresponsive for a moment (finishing to prepare) if you open it too quickly.

### web_app/sketchy

To launch the app, run

```bash
cd web_app/sketchy
npm start
```

the web application should open in your brower at http://localhost:3000/
