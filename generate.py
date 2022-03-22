import torch

from model_params import *
from wtv2_constants import *
from generation_params import *


def generate(epoch, tokenizer, model, device, loader, chosen_model_params, no_samples=None, gen_params=None, verbose=True):
    if gen_params is None:
        gen_params = default_gen_params

    model.eval()

    predictions = []
    actuals = []
    questions = []

    with torch.no_grad():
        for _, data in enumerate(loader, start=0):
            source_ids = data["source_ids"].to(device, dtype=torch.long)
            source_mask = data["source_mask"].to(device, dtype=torch.long)
            target_ids = data["target_ids"].to(device, dtype=torch.long)

            # At inference time, it is recommended to use generate(). This method takes care of encoding
            # the input and feeding the encoded hidden states via cross-attention layers to the decoder and
            # auto-regressively generates the decoder output
            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=chosen_model_params[MAX_TARGET_TEXT_LENGTH],
                num_beams=gen_params[BEAM_SIZE],
                repetition_penalty=gen_params[REPETITION_PENALTY],
                length_penalty=gen_params[LENGTH_PENALTY],
                early_stopping=gen_params[EARLY_STOPPING]
            )

            predicted_explanations = [
                tokenizer.decode(generated_id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                generated_id in generated_ids]
            actual_explanations = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                                   id in target_ids]
            inputs = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in
                      source_ids]
            if _ % 10 == 0 and verbose:
                for i in range(len(inputs)):
                    print("-------------------------------------------------")
                    print("SHOWING EXAMPLE:")
                    print("input:", inputs[i])
                    print("predicted_explanations:", predicted_explanations[i])
                    print("actual_explanations:", actual_explanations[i])

            predictions.extend(predicted_explanations)
            actuals.extend(actual_explanations)
            questions.extend(inputs)

            if no_samples is not None and no_samples >= _:
                break

    return questions, predictions, actuals


# todo: if you want to try u need to add gen params
def generate_with_inference_chains(epoch, tokenizer, model, device, loader, model_params):
    model.eval()

    predictions = []
    actuals = []
    questions = []

    with torch.no_grad():
        for _, data in enumerate(loader, start=0):
            source_ids = data["source_ids"].to(device, dtype=torch.long)
            source_mask = data["source_mask"].to(device, dtype=torch.long)
            target_ids = data["target_ids"].to(device, dtype=torch.long)

            actual_explanations = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                                   id in target_ids]
            actuals.extend(actual_explanations)

            input = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in
                     source_ids]
            questions.extend(input)
            batch_size = len(input)

            inference_step_to_generated = dict(zip([i for i in range(model_params[NO_INFERENCE_STEPS] + 1)],
                                                   [[] for _ in range(model_params[NO_INFERENCE_STEPS] + 1)]))

            # for first inference step
            sources = input
            generated = ["" for _ in range(batch_size)]
            for i in range(model_params[NO_INFERENCE_STEPS] + 1):
                sources, _, source_ids, source_mask = get_chain_source_ids_and_source_mask(
                    tokenizer=tokenizer,
                    max_len=model_params[MAX_SOURCE_TEXT_LENGTH],
                    sources_before=sources,
                    generated_before=generated,
                    # need sep becuz data set this will run on is the normal one without sep
                    separator=" $$ ",
                    retrieved=[]
                )
                source_ids = source_ids.to(device, dtype=torch.long)
                source_mask = source_mask.to(device, dtype=torch.long)

                generated_ids = model.generate(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    max_length=model_params[MAX_TARGET_TEXT_LENGTH],
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True
                )

                generated = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True)
                             for id in generated_ids]

                inference_step_to_generated[i].extend(generated)

            predicted_explanations = []
            for batch_index in range(batch_size):
                predicted_explanations.append(
                    " || ".join([inference_step_to_generated[inference_step][batch_index] for inference_step in
                                 range(model_params[NO_INFERENCE_STEPS] + 1)])
                )
            predictions.extend(predicted_explanations)

    return questions, predictions, actuals


# loader here has to contain a normal / not chained dataset
def generate_with_chains(epoch, tokenizer, model, device, loader, model_params, no_samples=None, gen_params=None, verbose=True):
    if gen_params is None:
        gen_params = default_gen_params

    model.eval()

    predictions = []
    actuals = []
    questions = []
    all_retrieved_central_facts = []
    all_retrieved_grounding_facts = []
    all_retrieved_lexglue_facts = []

    with torch.no_grad():
        for _, data in enumerate(loader, start=0):
            source_ids = data["source_ids"].to(device, dtype=torch.long)
            source_mask = data["source_mask"].to(device, dtype=torch.long)
            target_ids = data["target_ids"].to(device, dtype=torch.long)
            if model_params[AUGMENT_INPUT_WITH_RETRIEVED_FACTS]:
                central_retrieved = data[CENTRAL_RETRIEVED]
                all_retrieved_central_facts.extend(central_retrieved)
                grounding_retrieved = data[GROUNDING_RETRIEVED]
                all_retrieved_grounding_facts.extend(grounding_retrieved)
                lexglue_retrieved = data[LEXGLUE_RETRIEVED]
                all_retrieved_lexglue_facts.extend(lexglue_retrieved)
            else:
                central_retrieved = []
                grounding_retrieved = []
                lexglue_retrieved = []

            role_to_retrieved = {
                CENTRAL: central_retrieved,
                GROUNDING: grounding_retrieved,
                LEXGLUE: lexglue_retrieved
            }

            actual_explanations = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                                   id in target_ids]
            actuals.extend(actual_explanations)

            input = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in
                     source_ids]
            questions.extend(input)

            roles_order = [CENTRAL, GROUNDING, LEXGLUE] if model_params[CENTRAL_FIRST] else [GROUNDING, CENTRAL,
                                                                                             LEXGLUE]
            role_sources_without_retrieved = input
            empty_generated = ["" for _ in range(len(input))]
            role_generated = empty_generated
            central_generated = []
            grounding_generated = []
            lexglue_generated = []
            role_to_generated = {
                CENTRAL: central_generated,
                GROUNDING: grounding_generated,
                LEXGLUE: lexglue_generated
            }
            # todo: i think sources_before should be inputs when in NO_CHAIN_DEP mode
            for role in roles_order:
                role_sources, role_sources_without_retrieved, role_source_ids, role_source_mask = get_chain_source_ids_and_source_mask(
                    tokenizer=tokenizer,
                    max_len=model_params[MAX_SOURCE_TEXT_LENGTH],
                    sources_before=role_sources_without_retrieved,
                    generated_before=empty_generated if model_params[NO_CHAIN_DEP] else role_generated,
                    # need sep becuz data set this will run on is the normal one without sep
                    separator=explanatory_role_to_sep[role],
                    retrieved=role_to_retrieved[role]
                )
                role_source_ids = role_source_ids.to(device, dtype=torch.long)
                role_source_mask = role_source_mask.to(device, dtype=torch.long)

                role_generated_ids = model.generate(
                    input_ids=role_source_ids,
                    attention_mask=role_source_mask,
                    max_length=model_params[MAX_TARGET_TEXT_LENGTH],
                    num_beams=gen_params[BEAM_SIZE],
                    repetition_penalty=gen_params[REPETITION_PENALTY],
                    length_penalty=gen_params[LENGTH_PENALTY],
                    early_stopping=gen_params[EARLY_STOPPING]
                )
                role_generated = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True)
                                  for id in role_generated_ids]

                role_to_generated[role].extend(role_generated)

            predicted_explanations = []
            for i in range(len(input)):
                predicted_explanations.append(
                    " || ".join([central_generated[i], grounding_generated[i], lexglue_generated[i]])
                )
            predictions.extend(predicted_explanations)

            if no_samples is not None and no_samples >= _:
                break

    return questions, all_retrieved_central_facts, all_retrieved_grounding_facts, all_retrieved_lexglue_facts, predictions, actuals


def get_chain_source_ids_and_source_mask(tokenizer, max_len, sources_before, generated_before, separator, retrieved=[]):
    new_source_ids = []
    new_masks = []
    new_sources = []
    new_sources_without_retrieved = []

    for i in range(len(sources_before)):
        source_text_without_retrieval = sources_before[i] + generated_before[i] + separator
        source_text_with_retrieval = source_text_without_retrieval + (" @@ " + retrieved[i] if retrieved else "")

        # clean data, make sure it's a string
        source_text_without_retrieval = " ".join(source_text_without_retrieval.split())
        source_text_with_retrieval = " ".join(source_text_with_retrieval.split())

        new_sources.append(source_text_with_retrieval)
        new_sources_without_retrieved.append(source_text_without_retrieval)

        # tokenizing source
        source = tokenizer.batch_encode_plus(
            [source_text_with_retrieval],
            max_length=max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        source_ids = source["input_ids"].squeeze()
        source_ids = source_ids.to(dtype=torch.long)

        source_mask = source["attention_mask"].squeeze()
        source_mask = source_mask.to(dtype=torch.long)

        new_source_ids.append(source_ids)
        new_masks.append(source_mask)

    # these have to be tensors
    return new_sources, new_sources_without_retrieved, torch.stack(new_source_ids), torch.stack(new_masks)
