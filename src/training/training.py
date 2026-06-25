import torch
import os
import json
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from typing import Optional


class SmoLLMTrainer:
    def __init__(
        self,
        tokenizer: object,
        model: nn.Module,
        lr: float = 1e-4,
        ignore_index: int = -100,
        model_save_dir: str = "resources/SmoLLM-100M-Baby-LM-Base",
        save_model: bool = True,
        accumulation_steps: int = 1,
        save_every_n_steps: int = 250_000_000,
        warmup_steps: int = 2000,
        max_steps: int = 750_000,
    ):
        self.tokenizer = tokenizer
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.save_every_n_steps = save_every_n_steps
        self.save_model = save_model
        self.model = model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.loss_criteraion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.model_save_dir = model_save_dir

        self.history = {"train_loss": [], "eval_loss": []}

        now = datetime.now()

        self.formatted_date = now.strftime("%Y_%m_%d__%H_%M")

        self.accumulation_steps = accumulation_steps

    def train_epoch(
        self,
        dataloader,
        epoch_idx: int,
    ):
        self.model.train()
        total_loss = 0
        valid_batches = 0

        loop = tqdm(dataloader, desc=f"Epoch : {epoch_idx}")

        self.optimizer.zero_grad()

        for step, batch in enumerate(loop):
            input_ids = batch.get("input_ids").to(self.device)
            labels = batch.get("labels").to(self.device)

            logits = self.model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            batch_loss = self.loss_criteraion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                del logits, shift_logits, shift_labels, batch_loss
                continue

            loss = batch_loss / self.accumulation_steps
            loss.backward()

            if (step + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self._empty_cache()

            if self.save_model and (step + 1) % self.save_every_n_steps == 0:
                save_dir = self._save_model(step_idx=step + 1)
                tqdm.write(f"Checkpoint saved at step {step + 1} -> {save_dir}")

            total_loss += batch_loss.item()
            valid_batches += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            loop.set_postfix(
                loss=batch_loss.item(),
                lr=f"{current_lr:.6f}",
            )
            del logits, shift_logits, shift_labels, batch_loss, loss

        if valid_batches % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._empty_cache()

        return total_loss / valid_batches if valid_batches > 0 else float("inf")

    def eval_epoch(
        self,
        dataloader,
        epoch_idx: int,
    ):
        self.model.eval()
        total_loss = 0

        loop = tqdm(dataloader, desc=f"Val Epoch : {epoch_idx}")

        with torch.no_grad():
            for step, batch in enumerate(loop):
                input_ids = batch.get("input_ids").to(self.device)
                labels = batch.get("labels").to(self.device)

                logits = self.model(input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                batch_loss = self.loss_criteraion(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

                total_loss += batch_loss.item()

                loop.set_postfix(loss=batch_loss.item())

                del logits, shift_logits, shift_labels, batch_loss

                if step % 500 == 0:
                    self._empty_cache()

        return total_loss / len(dataloader)

    def _save_model(
        self,
        epoch_idx: Optional[int] = None,
        step_idx: Optional[int] = None,
    ):

        if epoch_idx:
            model_path = os.path.join(
                self.model_save_dir,
                f"epoch_{epoch_idx}",
                f"date_of_processing_{self.formatted_date}",
                "epochs",
                f"epoch_{epoch_idx}",
            )
        elif step_idx:
            model_path = os.path.join(
                self.model_save_dir,
                f"epoch_{epoch_idx}",
                f"date_of_processing_{self.formatted_date}",
                "steps",
                f"step_{step_idx}",
            )

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        final_model_file = os.path.join(model_path, "weights.pt")

        model_weights = self.model.state_dict()

        torch.save(model_weights, final_model_file)

        tokenizer_save_path = os.path.join(model_path, "tokenizer.json")
        self.tokenizer.tokenizer.save(tokenizer_save_path)

        return final_model_file

    def _save_metrics(
        self,
    ):
        metrics_path = os.path.join(self.model_save_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.history, f, indent=4)
        pass

    def _empty_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

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
            if self.save_model:
                save_path = self._save_model(epoch_idx=epoch)
                print(f"Epoch {epoch} saved in {save_path}")
        if self.save_model:
            self._save_metrics()
