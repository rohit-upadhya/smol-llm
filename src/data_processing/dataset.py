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
        self.eos_token_id = self.smollm_tokenizer.tokenizer.token_to_id("[EOS]")

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

        input_ids = input_ids[: self.max_length - 1] + [self.eos_token_id]

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
        self.eos_token_id = self.smollm_tokenizer.tokenizer.token_to_id("[EOS]")

    def __iter__(self):
        for item in self.dataset:
            text = item.get("text", "")

            input_ids = self.smollm_tokenizer.encode(text)

            input_ids = input_ids[: self.max_length - 1] + [self.eos_token_id]

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


class SmoLLMInstructDataset:
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
        self.eos_token_id = self.smollm_tokenizer.tokenizer.token_to_id("[EOS]")

    def __getitem__(
        self,
        key,
    ):
        example = self.dataset[key]

        prompt_str, response = self._format_prompt(example=example)

        prompt_ids = self.smollm_tokenizer.encode(prompt_str)

        response_ids = self.smollm_tokenizer.encode(response) + [self.eos_token_id]

        input_ids = prompt_ids + response_ids

        input_ids = input_ids[: self.max_length]

        labels = [-100] * len(prompt_ids) + response_ids

        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _format_prompt(
        self,
        example: dict,
    ):
        instruction = example.get("instruction", "")
        context = example.get("context", None)
        response = example.get("response", "")

        if context:
            user_message = f"[USER] {instruction} \n\n {context} [/USER]"
        else:
            user_message = f"[USER] {instruction} [/USER]"

        return (
            f"[SYSTEM] You are a helpful bot [/SYSTEM]\n"
            f"{user_message}\n"
            "[ASSISTANT] "
        ), response

    def __len__(self):
        return len(self.dataset)


def create_dateset(
    dataset,
    tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
    instruct: bool = False,
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
        if instruct:
            final_dataset = SmoLLMInstructDataset(
                dataset=dataset,
                tokenizer=tokenizer,
                max_length=max_length,
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
