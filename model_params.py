# todo make sense of these
MODEL = "MODEL"
TRAIN_BATCH_SIZE = "TRAIN_BATCH_SIZE"
VALID_BATCH_SIZE = "VALID_BATCH_SIZE"
TRAIN_EPOCHS = "TRAIN_EPOCHS"
VAL_EPOCHS = "VAL_EPOCHS"
LEARNING_RATE = "LEARNING_RATE"
MAX_SOURCE_TEXT_LENGTH = "MAX_SOURCE_TEXT_LENGTH"
MAX_TARGET_TEXT_LENGTH = "MAX_TARGET_TEXT_LENGTH"
SEED = "SEED"

t5_model_params = {
    "MODEL": "t5-base",
    "TRAIN_BATCH_SIZE": 4,
    "VALID_BATCH_SIZE": 4,
    "TRAIN_EPOCHS": 10,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 1e-3,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42
}

bart_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 4,
    "VALID_BATCH_SIZE": 4,
    "TRAIN_EPOCHS": 10,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 1e-3,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42
}