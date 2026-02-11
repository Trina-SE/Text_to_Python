"""
Configuration for all models and training.
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "Checkpoints")
ATTENTION_VIS_DIR = os.path.join(PROJECT_ROOT, "Attention_Visualizations")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "Evaluation")

# Dataset
DATASET_NAME = "Nan-Do/code-search-net-python"
NUM_TRAIN_EXAMPLES = 10000
NUM_VAL_EXAMPLES = 1500
NUM_TEST_EXAMPLES = 1500
MAX_SRC_LEN = 50   # docstring token limit
MAX_TRG_LEN = 80   # code token limit
MAX_SRC_LEN_EXTENDED = 100  # extended docstring token limit
MAX_TRG_LEN_EXTENDED = 150  # extended code token limit
FREQ_THRESHOLD = 2  # minimum word frequency for vocabulary

# Model hyperparameters
EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD = 1.0
SEED = 42

# Special token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
