import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "CCA"
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256
_C.INPUT.PRE_QUERY_SIZE = 300

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.CCA = CN()
_C.MODEL.CCA.NUM_CLIPS = 128

_C.MODEL.CCA.FEATPOOL = CN()
_C.MODEL.CCA.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.CCA.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.CCA.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.CCA.FEAT2D = CN()
_C.MODEL.CCA.FEAT2D.POOLING_COUNTS = [15,8,8,8]

_C.MODEL.CCA.INTEGRATOR = CN()
_C.MODEL.CCA.INTEGRATOR.QUERY_HIDDEN_SIZE = 512
_C.MODEL.CCA.INTEGRATOR.LSTM = CN()
_C.MODEL.CCA.INTEGRATOR.LSTM.NUM_LAYERS = 3
_C.MODEL.CCA.INTEGRATOR.LSTM.BIDIRECTIONAL = False

_C.MODEL.CCA.PREDICTOR = CN() 
_C.MODEL.CCA.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.CCA.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.CCA.PREDICTOR.NUM_STACK_LAYERS = 8

_C.MODEL.CCA.LOSS = CN()
_C.MODEL.CCA.LOSS.MIN_IOU = 0.3
_C.MODEL.CCA.LOSS.MAX_IOU = 0.7

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 12
_C.TEST.NMS_THRESH = 0.4
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
# cvse
_C.num_attribute = 300
_C.input_channel = 300
_C.embed_size = 1024
_C.adj_file = ''
_C.norm_func_type = ''
_C.inp_name = ''
_C.concept_name = ''
_C.com_concept = ''
_C.com_emb = ''
_C.num_path = ''