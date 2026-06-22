import torch
import os
import json
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime


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

        self.history = {"train_loss": [], "eval_loss": []}

        now = datetime.now()

        self.formatted_date = now.strftime("%Y_%m_%d__%H_%M")

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

        loop = tqdm(dataloader, desc=f"Val Epoch : {epoch_idx}")

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

        model_path = os.path.join(
            self.model_save_dir,
            f"date_of_processing_{self.formatted_date}___epoch_{epoch_idx}",
        )
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        final_model_file = os.path.join(model_path, "weights.pt")

        model_weights = self.model.state_dict()

        return torch.save(model_weights, final_model_file), final_model_file

    def _save_metrics(
        self,
    ):
        metrics_path = os.path.join(self.model_save_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.history, f, indent=4)
        pass

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
            avg_eval_loss = self.eval_epoch(dataloader=test_dataloader, epoch_idx=epoch)

            print(f"Average train loss for Epoch : {epoch} = {avg_train_loss}.")
            print(f"Average eval loss for Epoch : {epoch} = {avg_eval_loss}.")
            self.history["train_loss"].append(avg_train_loss)
            self.history["eval_loss"].append(avg_eval_loss)
            _, save_path = self._save_model(epoch_idx=epoch)
            print(f"Epoch {epoch} saved in {save_path}")

        self._save_metrics()
