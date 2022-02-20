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
AUGMENT_INPUT_WITH_RETRIEVED_FACTS = "AUGMENT_INPUT_WITH_RETRIEVED_FACTS"
NO_SIMILAR_HYPOTHESIS = "NO_SIMILAR_HYPOTHESIS"
NO_FACTS_TO_RETRIEVE = "NO_FACTS_TO_RETRIEVE"
QUESTION_AND_ANSWER = "question_and_answer"
HYPOTHESIS = "hypothesis"
TRAIN_ON = "TRAIN_ON"
ONLY_CETRAL = "ONLY_CETRAL"

t5_model_params = {
    "MODEL": "t5-base",
    "TRAIN_BATCH_SIZE": 4,
    "VALID_BATCH_SIZE": 4,
    "TRAIN_EPOCHS": 10,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 1e-3,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": True,
    "NO_SIMILAR_HYPOTHESIS": 3,
    "NO_FACTS_TO_RETRIEVE": 5,
    "ONLY_CETRAL": True,
    "TRAIN_ON": QUESTION_AND_ANSWER
}

bart_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 4,
    "VALID_BATCH_SIZE": 4,
    "TRAIN_EPOCHS": 10,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": True,
    "NO_SIMILAR_HYPOTHESIS": 20,
    "NO_FACTS_TO_RETRIEVE": 6,
    "ONLY_CETRAL": True,
    "TRAIN_ON": QUESTION_AND_ANSWER
}