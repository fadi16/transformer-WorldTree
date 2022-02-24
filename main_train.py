import pandas as pd
from train import trainer
from model_params import *
from retrieve_prompt_generate import retrieve
from main_eval import CENTRAL_FACTS_SEP, GROUNDING_FACTS_SEP, LEXGLUE_FACTS_SEP
from transformers import BartTokenizer, BartForConditionalGeneration
from wt_dataset import WorldTreeDataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, Adafactor
from transformers import T5Tokenizer, T5ForConditionalGeneration

####################### FIXED PATHS #############################
path_train = "./data/v2-proper-data/train_data_wed.csv"
path_dev = "./data/v2-proper-data/dev_data_wed.csv"
############################################################
path_train_chains = None
path_dev_chains = None

if __name__ == "__main__":

    ####################### CHANGE AS APPROPRRIATE #######################
    chosen_model_params = bart_chain_grounding_first_model_params
    for k, v in chosen_model_params.items():
        print(k, ":\t", v)
    ######################################################################

    if "bart" in chosen_model_params[MODEL]:
        # BART tokenizer
        tokenizer = BartTokenizer.from_pretrained(chosen_model_params[MODEL])
        # BART model for conditional generation
        model = BartForConditionalGeneration.from_pretrained(chosen_model_params[MODEL])
        optimizer = AdamW(
            model.parameters(),
            lr=3e-5,
        )
        print("***************** TRAINING BART *******************")
    else:  # T5
        # t5-plain tokenizer
        tokenizer = T5Tokenizer.from_pretrained(chosen_model_params[MODEL])
        # t5-plain model for conditional generation
        model = T5ForConditionalGeneration.from_pretrained(chosen_model_params[MODEL])
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
        print("***************** TRAINING T5 *******************")

    df_train = pd.read_csv(path_train, delimiter="\t")
    df_dev = pd.read_csv(path_dev, delimiter="\t")

    if chosen_model_params[CHAIN]:
        df_train_chains = pd.read_csv(chosen_model_params[TRAIN_CHAIN_CSV_PATH], delimiter="\t")
        df_dev_chains = pd.read_csv(chosen_model_params[DEV_CHAINS_CSV_PATH], delimiter="\t")

    if chosen_model_params[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
        if chosen_model_params[CHAIN]:
            print("USING RETRIEVAL METHOD - chain")
            central_train_retrieved_facts, central_dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                                           testing_df=df_dev,
                                                                                           no_similar_hypotheses=
                                                                                           chosen_model_params[
                                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                                           no_retrieved_facts=
                                                                                           chosen_model_params[
                                                                                               NO_FACTS_TO_RETRIEVE],
                                                                                           only_central=True,
                                                                                           retrieved_facts_sep=CENTRAL_FACTS_SEP)
            print("finished retrieving central facts")
            grounding_train_retrieved_facts, grounding_dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                                               testing_df=df_dev,
                                                                                               no_similar_hypotheses=
                                                                                               chosen_model_params[
                                                                                                   NO_SIMILAR_HYPOTHESIS],
                                                                                               no_retrieved_facts=
                                                                                               chosen_model_params[
                                                                                                   NO_FACTS_TO_RETRIEVE],
                                                                                               only_grounding=True,
                                                                                               retrieved_facts_sep=GROUNDING_FACTS_SEP)
            print("finished retrieving grounding facts")
            lexglue_train_retrieved_facts, lexglue_dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                                           testing_df=df_dev,
                                                                                           no_similar_hypotheses=
                                                                                           chosen_model_params[
                                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                                           no_retrieved_facts=
                                                                                           chosen_model_params[
                                                                                               NO_FACTS_TO_RETRIEVE],
                                                                                           only_lexglue=True,
                                                                                           retrieved_facts_sep=LEXGLUE_FACTS_SEP)
            print("finished retrieving lexglue facts")
            train_length = len(df_train_chains.index)
            dev_length = len(df_dev_chains.index)

            # todo FA: test
            for i, j in zip(range(0, train_length, 3), range(0, len(central_train_retrieved_facts))):
                df_train_chains[chosen_model_params[TRAIN_ON]][i] += " @@ " + (central_train_retrieved_facts[j] if chosen_model_params[CENTRAL_FIRST] else grounding_train_retrieved_facts[j])
                df_train_chains[chosen_model_params[TRAIN_ON]][i + 1] += " @@ " + (grounding_train_retrieved_facts[j] if chosen_model_params[CENTRAL_FIRST] else central_train_retrieved_facts[j])
                df_train_chains[chosen_model_params[TRAIN_ON]][i + 2] += " @@ " + lexglue_train_retrieved_facts[j]

            for i, j in zip(range(0, dev_length, 3), range(0, len(central_dev_retrieved_facts))):
                df_dev_chains[chosen_model_params[TRAIN_ON]][i] += " @@ " + (central_dev_retrieved_facts[j] if chosen_model_params[CENTRAL_FIRST] else grounding_dev_retrieved_facts[j])
                df_dev_chains[chosen_model_params[TRAIN_ON]][i + 1] += " @@ " + (grounding_dev_retrieved_facts[j] if chosen_model_params[CENTRAL_FIRST] else central_dev_retrieved_facts[j])
                df_dev_chains[chosen_model_params[TRAIN_ON]][i + 2] += " @@ " + lexglue_dev_retrieved_facts[j]
        else:
            print("USING RETRIEVAL METHOD - no chain")
            train_retrieved_facts, dev_retrieved_facts = retrieve.retrieve(training_df=df_train,
                                                                           testing_df=df_dev,
                                                                           no_similar_hypotheses=chosen_model_params[
                                                                               NO_SIMILAR_HYPOTHESIS],
                                                                           no_retrieved_facts=chosen_model_params[
                                                                               NO_FACTS_TO_RETRIEVE],
                                                                           only_central=chosen_model_params[
                                                                               ONLY_CETRAL])
            for i in range(len(train_retrieved_facts)):
                df_train[chosen_model_params[TRAIN_ON]][i] += " @@ " + train_retrieved_facts[i]
            for i in range(len(dev_retrieved_facts)):
                df_dev[chosen_model_params[TRAIN_ON]][i] += " @@ " + dev_retrieved_facts[i]

    source_text = chosen_model_params[TRAIN_ON]
    target_text = "explanation"

    df_train_chains = df_train_chains[[source_text, target_text]] if df_train_chains is not None else None
    df_train = df_train[[source_text, target_text]] if df_train is not None else None
    df_dev_chains = df_dev_chains[[source_text, target_text]] if df_dev_chains is not None else None
    df_dev = df_dev[[source_text, target_text]] if df_dev is not None else None

    training_dataset = WorldTreeDataset(
        dataframe=df_train_chains if chosen_model_params[CHAIN] else df_train,
        tokenizer=tokenizer,
        target_len=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
        source_len=chosen_model_params[MAX_SOURCE_TEXT_LENGTH],
        target_text_column_name=target_text,
        source_text_column_name=source_text
    )
    print(f"TRAINING Dataset: {df_train_chains.shape if df_train_chains is not None else df_train.shape}\n")
    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=chosen_model_params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0
    )

    validation_dataset = WorldTreeDataset(
        dataframe=df_dev_chains if chosen_model_params[CHAIN] else df_dev,
        tokenizer=tokenizer,
        target_len=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
        source_len=chosen_model_params[MAX_SOURCE_TEXT_LENGTH],
        target_text_column_name=target_text,
        source_text_column_name=source_text
    )
    print(f"VALIDATION Dataset: {df_dev_chains.shape if df_dev_chains is not None else df_dev.shape}\n")
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=chosen_model_params[VALID_BATCH_SIZE],
        shuffle=False,
        num_workers=0
    )

    if chosen_model_params[CHAIN]:
        df_dev = df_dev[[source_text, target_text]]
        validation_dataset2 = WorldTreeDataset(
            dataframe=df_dev,
            tokenizer=tokenizer,
            target_len=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
            source_len=chosen_model_params[MAX_SOURCE_TEXT_LENGTH],
            target_text_column_name=target_text,
            source_text_column_name=source_text,
            central_retrieved=central_dev_retrieved_facts if chosen_model_params[
                AUGMENT_INPUT_WITH_RETRIEVED_FACTS] else [],
            grounding_retrieved=grounding_dev_retrieved_facts if chosen_model_params[
                AUGMENT_INPUT_WITH_RETRIEVED_FACTS] else [],
            lexglue_retrieved=lexglue_dev_retrieved_facts if chosen_model_params[
                AUGMENT_INPUT_WITH_RETRIEVED_FACTS] else []
        )

        validation_loader2 = DataLoader(
            dataset=validation_dataset2,
            batch_size=chosen_model_params[VALID_BATCH_SIZE],
            shuffle=False,
            num_workers=0
        )
    else:
        validation_loader2 = None

    trainer(model=model, tokenizer=tokenizer, optimizer=optimizer, training_loader=training_loader,
            validation_loader=validation_loader,
            validation_loader2=validation_loader2, chosen_model_params=chosen_model_params,
            output_dir="./outputs")
