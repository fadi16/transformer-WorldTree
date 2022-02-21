import torch

from model_params import MAX_TARGET_TEXT_LENGTH, MAX_SOURCE_TEXT_LENGTH
from generate_v2_data import GROUNDING, LEXGLUE, CENTRAL, explanatory_role_to_sep


def validate(epoch, tokenizer, model, device, loader, model_params):
    # a switch for some kind of layers that behave differently during training and inference
    # common practise in evaluations is to use model.eval() with torch.no_grad() to turn off grad
    # computation during inference which speeds up computation cuz grads are not used for inference
    model.eval()

    predictions = []
    actuals = []

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
                max_length=model_params[MAX_TARGET_TEXT_LENGTH],
                num_beams=2,  # todo: how come?
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            predicted_explanations = [
                tokenizer.decode(generated_id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                generated_id in generated_ids]
            actual_explanations = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                                   id in target_ids]
            inputs = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in
                      source_ids]
            if _ % 10 == 0:
                for i in range(len(inputs)):
                    print("-------------------------------------------------")
                    print("SHOWING EXAMPLE:")
                    print("input:", inputs[i])
                    print("predicted_explanations:", predicted_explanations[i])
                    print("actual_explanations:", actual_explanations[i])
            predictions.extend(predicted_explanations)
            actuals.extend(actual_explanations)

    return predictions, actuals


# loader here has to contain a normal / not chained dataset
def validate_with_chains(epoch, tokenizer, model, device, loader, model_params):
    model.eval()

    predictions = []
    actuals = []

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

            # source_ids_central = source_ids + central_separator
            central_sources, central_source_ids, central_source_mask = get_chain_source_ids_and_source_mask(
                tokenizer=tokenizer,
                max_len=model_params[MAX_SOURCE_TEXT_LENGTH],
                sources_before=input,
                generated_before=["", "", ""],
                separator=explanatory_role_to_sep[CENTRAL])

            central_generated_ids = model.generate(
                input_ids=central_source_ids,
                attention_mask=central_source_mask,
                max_length=model_params[MAX_TARGET_TEXT_LENGTH],
                num_beams=2,  # todo: how come?
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            central_generated = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id
                                 in central_generated_ids]

            # source_ids_grounding = source_ids_central + generated_ids_central + grounding_separator
            grounding_sources, grounding_source_ids, grounding_source_mask = get_chain_source_ids_and_source_mask(
                tokenizer=tokenizer,
                max_len=model_params[MAX_SOURCE_TEXT_LENGTH],
                sources_before=central_sources,
                generated_before=central_generated,
                separator=explanatory_role_to_sep[GROUNDING])

            grounding_generated_ids = model.generate(
                input_ids=grounding_source_ids,
                attention_mask=grounding_source_mask,
                max_length=model_params[MAX_TARGET_TEXT_LENGTH],
                num_beams=2,  # todo: how come?
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            grounding_generated = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for
                                   id in grounding_generated_ids]

            # source_ids_lexglue = source_ids_grounding + generated_ids_grounding + lexglue_separtor
            lexglue_sources, lexglue_source_ids, lexglue_source_mask = get_chain_source_ids_and_source_mask(
                tokenizer=tokenizer,
                max_len=model_params[MAX_SOURCE_TEXT_LENGTH],
                sources_before=grounding_sources,
                generated_before=grounding_generated,
                separator=explanatory_role_to_sep[LEXGLUE])

            lexglue_generated_generated_ids = model.generate(
                input_ids=lexglue_source_ids,
                attention_mask=lexglue_source_mask,
                max_length=model_params[MAX_TARGET_TEXT_LENGTH],
                num_beams=2,  # todo: how come?
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            lexglue_generated = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id
                                 in lexglue_generated_generated_ids]

            predicted_explanations = []
            for i in range(len(input)):
                predicted_explanations.append(
                    " || ".join([central_generated[i], grounding_generated[i], lexglue_generated[i]])
                )
            predictions.extend(predicted_explanations)

    return predictions, actuals


def get_chain_source_ids_and_source_mask(tokenizer, max_len, sources_before, generated_before, separator):
    new_source_ids = []
    new_masks = []
    new_sources = []

    for i in range(len(sources_before)):
        sources_before[i] += generated_before[i] + separator

        source_text = str(sources_before[i])

        # clean data, make sure it's a string
        source_text = " ".join(source_text.split())

        new_sources.append(source_text)

        # tokenizing source
        source = tokenizer.batch_encode_plus(
            [source_text],
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
    return new_sources, torch.stack(new_source_ids), torch.stack(new_masks)
