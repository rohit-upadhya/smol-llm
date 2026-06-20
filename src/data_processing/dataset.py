import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class SmoLLMDataLoader:
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 512,
    ):
        self.dataset = hf_dataset
        self.smollm_tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self.smollm_tokenizer.tokenizer.token_to_id("[PAD]")

    def __len__(
        self,
    ):
        return len(self.dataset)

    def __getitem__(
        self,
        idx,
    ):
        text = self.dataset["train"][idx]["text"]

        input_ids = self.smollm_tokenizer.encode(text)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        tensor_ids = torch.tensor(input_ids)

        return {"input_ids": tensor_ids, "labels": tensor_ids.clone()}


class SmoLLMCollate:
    def __init__(
        self,
        pad_token_id: int,
    ):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch,
    ):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids_padded = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )

        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
        }


def create_dateset(
    hf_dataset,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
):
    dataset = SmoLLMDataLoader(
        hf_dataset=hf_dataset, tokenizer=tokenizer, max_length=max_length
    )
    collate_fn = SmoLLMCollate(pad_token_id=dataset.pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
