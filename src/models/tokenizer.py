from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BPETrainer
from tokenizers.pre_tokenizer import Whitespace
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
        file_paths: list[str],
    ):
        trainer = BPETrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )
        self.tokenizer.train(files=file_paths, trainer=trainer)
        self.tokenizer.save(self.model_path)

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
