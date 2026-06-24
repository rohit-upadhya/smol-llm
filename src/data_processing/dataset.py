import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence


class SmoLLMDataLoader:
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 512,
        pad_token_id: str = "[PAD]",
    ):
        self.dataset = dataset
        self.smollm_tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = self.smollm_tokenizer.tokenizer.token_to_id(pad_token_id)

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

        tensor_ids = torch.tensor(input_ids, dtype=torch.long)

        return {"input_ids": tensor_ids, "labels": tensor_ids.clone()}


class SmoLLMIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 512,
    ):
        self.dataset = dataset
        self.smollm_tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for item in self.dataset:
            text = item.get("text", "")

            input_ids = self.smollm_tokenizer.encode(text)

            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]

            tensor_ids = torch.tensor(input_ids, dtype=torch.long)

            yield {"input_ids": tensor_ids, "labels": tensor_ids.clone()}


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
    dataset,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
):
    pad_id = tokenizer.tokenizer.token_to_id("[PAD]")
    collate_fn = SmoLLMCollate(pad_token_id=pad_id)

    try:
        _ = len(dataset)
        is_stream = False
    except TypeError:
        is_stream = True

    if is_stream:
        final_dataset = SmoLLMIterableDataset(
            dataset=dataset, tokenizer=tokenizer, max_length=max_length
        )
        return DataLoader(
            final_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        final_dataset = SmoLLMDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            pad_token_id="[PAD]",
        )
        return DataLoader(
            final_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
