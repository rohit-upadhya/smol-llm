from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


class SmoLMTokenizer:
    def __init__(
        self,
        vocab_size: int = 32000,
        model_path: str = "resources/smolllm_tokenizer.json",
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path

        if os.path.exists(self.model_path):
            self.tokenizer = HFTokenizer.from_file(self.model_path)
            print("Loaded Tokenizer")
        else:
            self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()

    def train(
        self,
        hf_dataset,
    ):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )

        def batch_iterator(
            batch_size=10000,
        ):
            for i in range(0, len(hf_dataset), batch_size):
                yield hf_dataset[i : i + batch_size]["text"]
            pass

        self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        self.tokenizer.save(self.model_path)
        print(f"Tokenizer saved to {self.model_path}.")

    def encode(
        self,
        text: str,
    ) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        return self.tokenizer.decode(token_ids)

    def load_or_train(
        self,
        hf_dataset,
    ):
        if not os.path.exists(self.model_path):
            print("Printing tokenizer.")
            self.train(hf_dataset=hf_dataset)
        print("Tokenizer loaded.")
