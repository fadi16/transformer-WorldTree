import torch

from model_params import MAX_TARGET_TEXT_LENGTH


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
                num_beams=2, # todo: how come?
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )

            predicted_explanations = [tokenizer.decode(generated_id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for generated_id in generated_ids]
            actual_explanations = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in target_ids]
            inputs = [tokenizer.decode(id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for id in source_ids]
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