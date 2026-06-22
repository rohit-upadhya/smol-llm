import torch
import os
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


class SmoLLMTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        ignore_index: int = -100,
        model_save_dir: str = "resources/SmoLLM-100M-Baby-LM-Base",
    ):

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.loss_criteraion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.model_save_dir = model_save_dir

    def train_epoch(
        self,
        dataloader,
        epoch_idx: int,
    ):
        self.model.train()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch : {epoch_idx}")

        for batch in loop:
            input_ids = batch.get("input_ids").to(self.device)
            labels = batch.get("labels").to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            epoch_loss = self.loss_criteraion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            epoch_loss.backward()
            self.optimizer.step()

            total_loss += epoch_loss.item()

            loop.set_postfix(loss=epoch_loss.item())

        return total_loss / len(dataloader)

    def eval_epoch(
        self,
        dataloader,
        epoch_idx: int,
    ):
        self.model.eval()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Epoch : {epoch_idx}")

        with torch.no_grad():
            for batch in loop:
                input_ids = batch.get("input_ids").to(self.device)
                labels = batch.get("labels").to(self.device)

                logits = self.model(input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                epoch_loss = self.loss_criteraion(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                total_loss += epoch_loss.item()

                loop.set_postfix(loss=epoch_loss.item())

        return total_loss / len(dataloader)

    def _save_model(
        self,
        epoch_idx,
    ):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        model_path = os.path.join(self.model_save_dir, f"epoch_{epoch_idx}_weights.pt")

        model_weights = self.model.state_dict()

        return torch.save(model_weights, model_path), model_path

    def fit(
        self,
        train_dataloader,
        test_dataloader,
        epochs: int = 3,
    ):
        for epoch in range(1, epochs + 1):
            avg_train_loss = self.train_epoch(
                dataloader=train_dataloader, epoch_idx=epoch
            )
            print(f"Average train loss for Epoch : {epoch} = {avg_train_loss}.")
            avg_eval_loss = self.eval_epoch(dataloader=test_dataloader, epoch_idx=epoch)
            print(f"Average eval loss for Epoch : {epoch} = {avg_eval_loss}.")

            _, save_path = self._save_model(epoch_idx=epoch)
