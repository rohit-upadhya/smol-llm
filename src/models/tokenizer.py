from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre, ByteLevel, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDec
import os


class SmoLLMTokenizer:
    def __init__(
        self,
        vocab_size: int = 32000,
        model_path: str = "resources/smolllm_tokenizer.json",
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path

        if os.path.exists(self.model_path):
            self.tokenizer = HFTokenizer.from_file(self.model_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
            print("Loaded Tokenizer")
        else:
            self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
            self.tokenizer.decoder = ByteLevelDec()

    def train(
        self,
        hf_dataset,
        num_docs_for_training: int = 3_500_000,
    ):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[EOS]"],
            initial_alphabet=ByteLevel.alphabet(),
        )

        def batch_iterator():
            for i, item in enumerate(hf_dataset):
                if i >= num_docs_for_training:
                    break
                yield item["text"]

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
