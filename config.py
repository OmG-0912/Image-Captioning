import os

# Paths
IMAGE_DIR = 'Images'
CAPTIONS_PATH = 'captions.txt'
FEATURES_PATH = 'features.pkl'
MODEL_SAVE_PATH = 'best_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'
LOG_DIR = 'logs'

# Model Hyperparameters
EMBEDDING_DIM = 256
DECODER_UNITS = 256
ATTENTION_UNITS = 256
DENSE_UNITS = 256
IMAGE_FEATURE_DIM = 2048  # Output of InceptionV3 (pooling='avg')

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
VAL_SPLIT = 0.1
RANDOM_STATE = 42

# Preprocessing
MIN_WORD_FREQ = 1  # Minimum word frequency to keep in vocabulary
MAX_CAPTION_LENGTH = 35  # Max length of sequences (startseq ... endseq)
