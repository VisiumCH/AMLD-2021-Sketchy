# Documentation per folder

First, always make sure that the file `options.py` contains the right metadata and constants for the task.

##### Replacing _dataset_ by one of sketchy, tuberlin, sk+tu, quickdraw, sk+tu+qd.

## data/

### loader_factory.py

To check that all dataset are loading well, you can call

```bash
python src/data/loader_factory.py *dataset*
```

It will load every dataset and print their length. Also, it creates an example image of the positive and negative images created for training. It can be seen in the src/visualization folder.

The available datasets are:

- Sketchy: for sketchy dataset (argument sketchy)
- TU-Berlin: for TU-Berlin dataset (argument tuberlin)
- Quickdraw: for Quickdraw dataset (argument quickdraw)
- Sk+Tu: for combining Sketchy and TU-Berlin datasets (argument sk+tu)
- Sk+Tu+Qd: for combining Sketchy, TU-Berlin and Quickdraw datasets (argument sk+tu+qd)

---

## models/

### train.py

To train a model on a dataset, verify options.py for your settings, then

```bash
python src/models/train.py *name*
```

the model and logs will be saved in a folder called name in io/models/name/.

### test.py

To test a dataset on a model,

```bash
python src/models/test.py *name*
```

### inference

The inference needs two steps.

1. First, the embeddings of all images must be pre-computed and save with their metadata. Therefore:

```bash
python src/models/inference/preprocess_embeddings.py *name*
```

the embeddings will be saved in the same folder as the model folder in a folder called precomputed_embeddings.

2. Then,

```bash
python src/models/inference/inference.py *name*
```

selects 20 random sketches. For each sketch, computes the embedding and the 4 closest embeddings of images. It then plots the sketches and the associated close images and saves them in the same folder as the model folder in a folder called predictions.

---

## visualization/

### visualize.py

---

## api/

### server.py
