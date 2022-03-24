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
CHAIN = "CHAIN"

TRAIN_CHAIN_CSV_PATH = "TRAIN_CHAIN_CSV_PATH"
DEV_CHAINS_CSV_PATH = "DEV_CHAINS_CSV_PATH"
DEV_CSV_PATH = "DEV_CSV_PATH"
TEST_CSV_PATH = "TEST_CSV_PATH"

CENTRAL_FIRST = "CENTRAL_FIRST"

NO_CHAIN_DEP = "NO_CHAIN_DEP"

CHAIN_ON = "CHAIN_ON"

PREVIOUS_SORTED = "PREVIOUS_SORTED"
ROLE = "ROLE"
NO_INFERENCE_STEPS = "NO_INFERENCE_STEPS"

NO_FACTS_TO_RETRIEVE_CENTRAL = "NO_FACTS_TO_RETRIEVE_CENTRAL"
NO_FACTS_TO_RETRIEVE_GROUNDING = "NO_FACTS_TO_RETRIEVE_GROUNDING"
NO_FACTS_TO_RETRIEVE_LEXGLUE = "NO_FACTS_TO_RETRIEVE_LEXGLUE"

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
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": False
}

bart_plain_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 4,
    "VALID_BATCH_SIZE": 4,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": False,
    "NO_SIMILAR_HYPOTHESIS": 0,
    "NO_FACTS_TO_RETRIEVE": 0,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": False
}

bart_retrieve_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 4,
    "VALID_BATCH_SIZE": 4,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": True,
    "NO_SIMILAR_HYPOTHESIS": 20,
    "NO_FACTS_TO_RETRIEVE": 6,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": False
}

bart_chain_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 6,
    "VALID_BATCH_SIZE": 12,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": False,
    "NO_SIMILAR_HYPOTHESIS": 0,
    "NO_FACTS_TO_RETRIEVE": 0,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": True,
    "TRAIN_CHAIN_CSV_PATH": "data/v2-proper-data/train_data_wed_chains.csv",
    "DEV_CHAINS_CSV_PATH": "data/v2-proper-data/dev_data_wed_chains.csv",
    "DEV_CSV_PATH": "data/v2-proper-data/dev_data_wed.csv",
    "NO_CHAIN_DEP": False,
    CHAIN_ON: ROLE,
    CENTRAL_FIRST: True
}

bart_chain_retrieve_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 6,
    "VALID_BATCH_SIZE": 6,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 280,
    "MAX_TARGET_TEXT_LENGTH": 280,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": True,
    "NO_SIMILAR_HYPOTHESIS": 20,
    "NO_FACTS_TO_RETRIEVE": 3,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": True,
    "TRAIN_CHAIN_CSV_PATH": "data/v2-proper-data/train_data_wed_chains.csv",
    "DEV_CHAINS_CSV_PATH": "data/v2-proper-data/dev_data_wed_chains.csv",
    "DEV_CSV_PATH": "data/v2-proper-data/dev_data_wed.csv",
    "CENTRAL_FIRST": True,
    "NO_CHAIN_DEP": False,
    CHAIN_ON: ROLE,
    NO_FACTS_TO_RETRIEVE_CENTRAL: 3,
    NO_FACTS_TO_RETRIEVE_GROUNDING: 3,
    NO_FACTS_TO_RETRIEVE_LEXGLUE: 3
}

# retrieve differentn number of facts for central etc
bart_chain_retrieve_different_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 6,
    "VALID_BATCH_SIZE": 6,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 280,
    "MAX_TARGET_TEXT_LENGTH": 280,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": True,
    "NO_SIMILAR_HYPOTHESIS": 20,
    "NO_FACTS_TO_RETRIEVE": 3,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": True,
    "TRAIN_CHAIN_CSV_PATH": "data/v2-proper-data/train_data_wed_chains.csv",
    "DEV_CHAINS_CSV_PATH": "data/v2-proper-data/dev_data_wed_chains.csv",
    "DEV_CSV_PATH": "data/v2-proper-data/dev_data_wed.csv",
    "CENTRAL_FIRST": True,
    "NO_CHAIN_DEP": False,
    CHAIN_ON: ROLE,
    NO_FACTS_TO_RETRIEVE_CENTRAL: 3,
    NO_FACTS_TO_RETRIEVE_GROUNDING: 2,
    NO_FACTS_TO_RETRIEVE_LEXGLUE: 1
}

bart_chain_grounding_first_model_params = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 6,
    "VALID_BATCH_SIZE": 6,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": False,
    "NO_SIMILAR_HYPOTHESIS": 0,
    "NO_FACTS_TO_RETRIEVE": 0,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": True,
    "TRAIN_CHAIN_CSV_PATH": "data/v2-proper-data/train_data_wed_chains_grounding_first.csv",
    "DEV_CHAINS_CSV_PATH": "data/v2-proper-data/dev_data_wed_chains_grounding_first.csv",
    "DEV_CSV_PATH": "data/v2-proper-data/dev_data_wed.csv",
    "CENTRAL_FIRST": False,
    "NO_CHAIN_DEP": False,
    "CHAIN_ON": ROLE
}

bart_chain_no_dep = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 6,
    "VALID_BATCH_SIZE": 6,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": False,
    "NO_SIMILAR_HYPOTHESIS": 0,
    "NO_FACTS_TO_RETRIEVE": 0,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": True,
    "TRAIN_CHAIN_CSV_PATH": "data/v2-proper-data/train_data_wed_chains_no_dep.csv",
    "DEV_CHAINS_CSV_PATH": "data/v2-proper-data/dev_data_wed_chains_no_dep.csv",
    "DEV_CSV_PATH": "data/v2-proper-data/dev_data_wed.csv",
    "CENTRAL_FIRST": True,
    "NO_CHAIN_DEP": True,
    "CHAIN_ON": ROLE
}

bart_chain_inference_steps = {
    "MODEL": "facebook/bart-base",
    "TRAIN_BATCH_SIZE": 10,
    "VALID_BATCH_SIZE": 20,
    "TRAIN_EPOCHS": 15,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 3e-5,
    "MAX_SOURCE_TEXT_LENGTH": 256,
    "MAX_TARGET_TEXT_LENGTH": 256,
    "SEED": 42,
    "AUGMENT_INPUT_WITH_RETRIEVED_FACTS": False,
    "NO_SIMILAR_HYPOTHESIS": 0,
    "NO_FACTS_TO_RETRIEVE": 0,
    "ONLY_CETRAL": False,
    "TRAIN_ON": QUESTION_AND_ANSWER,
    "CHAIN": True,
    "TRAIN_CHAIN_CSV_PATH": "data/v2-proper-data/train_data_wed_inference_chains_4.csv",
    "DEV_CHAINS_CSV_PATH": "data/v2-proper-data/dev_data_wed_inference_chains_4.csv",
    "DEV_CSV_PATH": "data/v2-proper-data/dev_data_wed.csv",
    "CENTRAL_FIRST": False,
    "NO_CHAIN_DEP": False,
    "CHAIN_ON": PREVIOUS_SORTED,
    "NO_INFERENCE_STEPS": 4
}
