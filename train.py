# generic for training
import torch
import rich
def train(epoch, tokenizer, model, device, loader, optimizer, logger):
    model.train()

    # todo: data should be a batch of inputs
    for _, data  in enumerate(loader, start=0):
        y = data["target_ids"].to(device, dtype=torch.long)
        # todo: what are these?
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        # In addition, we must make sure that padding token id’s of the labels are not taken into account by the loss function.
        # In PyTorch and Tensorflow, this can be done by replacing them with -100, which is the ignore_index of the CrossEntropyLoss
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)


        # todo: difference between lm_labels and y_ids
        outputs = model(
            input_ids=ids,  # ok
            attention_mask=mask,    # ok
            decoder_input_ids=y_ids, # todo this is not needed according to the documentation
            labels=lm_labels
        )

        # ids = data["source_ids"].to(device, dtype=torch.long)
        # mask = data["source_mask"].to(device, dtype=torch.long)
        # y = data["target_ids"].to(device, dtype=torch.long)
        # labels = y[:, 1:].clone().detach()
        # labels[y[:,1] == tokenizer.pad_token_id] = -100

        # outputs = model(
        #     input_ids=ids,
        #     attention_mask=mask,
        #     labels=labels
        # )

        # FA: this is cross entropy loss between predicted and golden output
        loss = outputs[0]

        if _ % 100 == 0:
            logger.add_row(str(epoch), str(_), str(loss))
            rich.print(logger)

        # clears old gradients from last step - so that they do not accumulate everytime you do loss.backwards
        optimizer.zero_grad()
        # back propagations
        loss.backward()
        # gradient decent
        optimizer.step()