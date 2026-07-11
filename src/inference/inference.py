import os
import torch
import torch.nn.functional as F
from typing import Optional
from src.models.models import SmoLLM
from src.models.tokenizer import SmoLLMTokenizer


class Inference:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_path: Optional[str] = None,
        n_heads: int = 12,
        dim: int = 768,
        n_layers: int = 12,
    ):
        self.model_name_or_path = model_name_or_path
        self.tokenizer_path = self.model_name_or_path
        if tokenizer_path:
            self.tokenizer_path = tokenizer_path
        self.n_heads = n_heads
        self.dim = dim
        self.n_layers = n_layers

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._init_model_tokenizer()

    def _init_model_tokenizer(
        self,
    ):
        self.tokenizer = SmoLLMTokenizer(model_path=self.tokenizer_path)
        self.llm = SmoLLM(
            vocab_size=self.tokenizer.vocab_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim=self.dim,
        ).to(self.device)

        print("Loading weights.")

        weights = torch.load(
            self.model_name_or_path, map_location=self.device, weights_only=True
        )

        self.llm.load_state_dict(weights)

        self.llm.eval()

        print("Model loaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repitition_penalty: float = 1.2,
    ):
        input_ids = self.tokenizer.encode(text=prompt)

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        eos_token_id = self.tokenizer.tokenizer.token_to_id("[EOS]")

        with torch.no_grad():
            for _ in range(max_tokens):

                if input_tensor.shape[1] >= 512:
                    break

                logits = self.llm(input_tensor)

                next_token_logits = logits[:, -1, :]

                if repitition_penalty != 1.0:
                    for tok in set(input_tensor[0].tolist()):
                        s = next_token_logits[:, tok]

                        next_token_logits[:, tok] = torch.where(
                            s < 0, s * repitition_penalty, s / repitition_penalty
                        )
                        pass

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(
                        next_token_logits, min(top_k, next_token_logits.size(-1))
                    )
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")

                if top_p < 1.0:
                    sl, si = torch.sort(next_token_logits, descending=True)
                    cummulative = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    remove = cummulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    next_token_logits[0, si[0, remove[0]]] = -float("inf")

                probs = F.softmax(next_token_logits, dim=-1)

                if temperature > 0:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                input_tensor = torch.cat((input_tensor, next_token), dim=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        generated_ids = input_tensor[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def probe_eos(self, prompt: str, max_tokens: int = 50):
        input_ids = self.tokenizer.encode(text=prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        eos_token_id = self.tokenizer.tokenizer.token_to_id("[EOS]")

        print(f"EOS id: {eos_token_id}")
        with torch.no_grad():
            for step in range(max_tokens):
                logits = self.llm(input_tensor)
                next_token_logits = logits[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)  # raw, unfiltered

                eos_prob = probs[0, eos_token_id].item()
                rank = (probs[0] > eos_prob).sum().item()  # how many tokens beat EOS

                next_token = torch.argmax(probs, dim=-1, keepdim=True)
                input_tensor = torch.cat((input_tensor, next_token), dim=1)

                print(
                    f"step {step:3d} | EOS prob: {eos_prob:.2e} | EOS rank: {rank}/{probs.size(-1)}"
                )

                if next_token.item() == eos_token_id:
                    print(">>> model chose EOS")
                    break


if __name__ == "__main__":
    WEIGHTS_PATH = "resources/SmoLLM-100M-Instruct_test/run_2026_07_11__18_22/checkpoint-epoch-3/pytorch_model.bin"
    TOKENIZER_PATH = "resources/SmoLLM-100M-Instruct_test/run_2026_07_11__18_22/checkpoint-epoch-3/tokenizer.json"
    if os.path.exists(WEIGHTS_PATH):
        inferencer = Inference(
            model_name_or_path=WEIGHTS_PATH, tokenizer_path=TOKENIZER_PATH
        )

        prompt = """[SYSTEM] You are a helpful bot [/SYSTEM]\n[USER] What is machine learning? [/USER]\n[ASSISTANT] """
        output = inferencer.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.8,
            top_p=0.90,
            repitition_penalty=1.2,
            top_k=50,
        )
        # test = inferencer.probe_eos("Machine Learning is ")
        # inferencer.probe_eos("The capital of France is ", max_tokens=100)
        # inferencer.probe_eos("Water is made of hydrogen and ")
        print("\n" + output)
    else:
        print(f"\tWeights arenot here : {WEIGHTS_PATH}")
        print("Check the directory.")
    pass
