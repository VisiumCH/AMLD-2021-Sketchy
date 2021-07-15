# ---------------- DATASETS -------------------
PROCESSED_PATH = "io/data/processed/"
SKETCHY = 'sketchy'
TUBERLIN = 'tuberlin'
QUICKDRAW = 'quickdraw'
SKTU = 'sk+tu'
SKTUQD = 'sk+tu+qd'
DATASETS = [SKETCHY, TUBERLIN, QUICKDRAW, SKTU, SKTUQD]

FOLDERS = {
    SKETCHY: "Sketchy",
    TUBERLIN: "TU-Berlin",
    QUICKDRAW: "Quickdraw"
}


# ---------------- INFERENCE ------------------
MODELS_PATH = "io/models/"
PREDICTION = "predictions"
EMBEDDINGS = "precomputed_embeddings"
DICT_CLASS = "_dict_class.json"
EMB_ARRAY = "_array.npy"
METADATA = "_meta.csv"
PARAMETERS = 'params.txt'
NUM_CLOSEST_PLOT = 4
NUMBER_RANDOM_IMAGES = 20
