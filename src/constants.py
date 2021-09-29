# ----------------- IO PATH -------------------
MODELS_PATH = "io/models/"
DATA_PATH = "io/data/raw/"

# ---------------- DATASETS -------------------
PROCESSED_PATH = "io/data/processed/"
SKETCHY = "sketchy"
TUBERLIN = "tuberlin"
QUICKDRAW = "quickdraw"
SKTU = "sk+tu"
SKTUQD = "sk+tu+qd"
DATASETS = [SKETCHY, TUBERLIN, QUICKDRAW, SKTU, SKTUQD]

FOLDERS = {SKETCHY: "Sketchy", TUBERLIN: "TU-Berlin", QUICKDRAW: "Quickdraw"}


# ---------------- INFERENCE ------------------
PREDICTION = "predictions"
EMBEDDINGS = "precomputed_embeddings"
DICT_CLASS = "_dict_class.json"
EMB_ARRAY = "_array.npy"
METADATA = "_meta.csv"
PARAMETERS = "params.txt"
NUM_CLOSEST_PLOT = 4
NUMBER_RANDOM_IMAGES = 20


# ----------------- SERVER ------------------
TENSORBOARD_IMAGE = "sprite.png"
TENSORBOARD_EMBEDDINGS = "tensors.tsv"
TENSORBOARD_CLASSES = "metadata.tsv"
NB_DATASET_IMAGES = 5
CUSTOM_SKETCH_CLASS = "My Custom Sketch"  # if modified, must change in web app as well

I_CROP_LEFT = 210
I_CROP_RIGHT = 1850
I_CROP_TOP = 300
I_CROP_BOTTOM = 670
INFERENCE_CROP = [I_CROP_LEFT, I_CROP_TOP, I_CROP_RIGHT, I_CROP_BOTTOM]

A_CROP_LEFT = 220
A_CROP_RIGHT = 1820
A_CROP_TOP = 140
A_CROP_BOTTOM = 680
ATTENTION_CROP = [A_CROP_LEFT, A_CROP_TOP, A_CROP_RIGHT, A_CROP_BOTTOM]
