import torch.nn as nn

from src.models.layers import RMSNorm

from src.models.transformer import TransformerBlock


class SmoLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_heads: int,
        dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.dim = dim
        self.n_layers = n_layers

        self.tok_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, n_heads=n_heads) for _ in range(n_layers)]
        )

        self.rms_norm_final_layer = RMSNorm(
            dim=dim,
        )

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids,
    ):
        # shape of input_ids is (Batch, Sequence_Length)
        x = self.tok_embeddings(input_ids)

        for item in self.transformer_blocks:
            x = item(x)

        x = self.rms_norm_final_layer(x)

        logits = self.lm_head(x)

        return logits
