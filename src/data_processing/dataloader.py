import torch
from torch.utils.data import Dataset, DataLoader


class SmolLMDataLoader:
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
        text = self.dataset[idx]["text"]

        input_ids = self.smollm_tokenizer.encode(text)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        else:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length

        tensor_ids = torch.tesor(input_ids)

        return {"input_ids": tensor_ids, "label": tensor_ids.clone()}


def create_dateset(
    hf_dataset,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
):
    dataset = SmolLMDataLoader(
        hf_dataset=hf_dataset, tokenizer=tokenizer, max_length=max_length
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
