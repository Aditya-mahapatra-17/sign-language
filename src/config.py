import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')

# Model Parameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# Class Labels
NUM_CLASSES = 29

# Inference & Smoothing Parameters (Threshold set to 0.30 so predictions display even at lower certainty)
CONFIDENCE_THRESHOLD = 0.30
SMOOTHING_WINDOW = 6
PREDICTION_COOLDOWN = 1.0 # seconds between accepting the same letter

# Logging level
LOG_LEVEL = 'INFO'
