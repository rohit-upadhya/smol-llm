from typing import Optional
from datasets import load_dataset

from src.data_processing.dataset import create_dateset
from src.models.tokenizer import SmoLLMTokenizer
from src.data_processing.hf_data_download import DataDownload
from src.data_processing.dataset import SmoLLMInstructDataset, SmoLLMCollate
from src.models.models import SmoLLM
from src.training.training import SmoLLMTrainer


class SmoLLMRunner:
    def _execute_pipeline(
        self,
        train_data,
        test_data,
        n_heads: int,
        dim: int,
        n_layers: int,
        lr: float,
        epochs: int,
        batch_size: int = 4,
        save_model: bool = True,
        accumulation_steps: int = 1,
        save_every_n_steps: int = 100_000,
        docs_to_take: int = 3_000_000,
        tokenizer_path: Optional[str] = None,
        model_save_dir: Optional[str] = None,
        upload_models: bool = True,
        checkpoint_path: Optional[str] = None,
        instruct: bool = False,
    ):
        self.batch_size = batch_size
        self.docs_to_take = docs_to_take
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        if tokenizer_path:
            tokenizer = SmoLLMTokenizer(model_path=tokenizer_path)
        else:
            tokenizer = SmoLLMTokenizer()
            tokenizer.load_or_train(hf_dataset=train_data)
        pad_token_id = tokenizer.tokenizer.token_to_id("[PAD]")

        if instruct:
            train_loader = create_dateset(
                train_data,
                tokenizer,
                batch_size=batch_size,
                shuffle=True,
                instruct=True,
            )
            test_loader = create_dateset(
                test_data,
                tokenizer,
                batch_size=batch_size,
                shuffle=False,
                instruct=True,
            )

            total_batches = len(train_loader)

        else:
            train_loader = create_dateset(
                dataset=train_data,
                tokenizer=tokenizer,
                shuffle=False,
                batch_size=batch_size,
            )

            test_loader = create_dateset(
                dataset=test_data,
                tokenizer=tokenizer,
                shuffle=True,
                batch_size=batch_size,
            )

            total_batches = self.docs_to_take // self.batch_size
        max_steps = (total_batches // self.accumulation_steps) * self.epochs
        warmup_steps = int(max_steps * 0.05)

        print(f"Number of Test Batches: {len(test_loader)}")

        model = SmoLLM(
            vocab_size=tokenizer.vocab_size,
            n_heads=n_heads,
            dim=dim,
            n_layers=n_layers,
        )
        self._print_model_size(model=model)
        trainer = SmoLLMTrainer(
            model=model,
            lr=lr,
            ignore_index=-100,
            save_model=save_model,
            accumulation_steps=accumulation_steps,
            save_every_n_steps=save_every_n_steps,
            tokenizer=tokenizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            model_save_dir=(
                model_save_dir
                if model_save_dir
                else "resources/SmoLLM-100M-Baby-LM-Base"
            ),
            upload_models=upload_models,
        )

        if checkpoint_path:
            trainer.load_checkpoints(chckpoint_path=checkpoint_path)
        trainer.fit(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            epochs=epochs,
        )

    def instruct_tune(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        model_save_dir: str,
        test_run: bool = False,
    ):
        raw = load_dataset("databricks/databricks-dolly-15k", split="train")
        split = raw.train_test_split(test_size=500, seed=42)
        train_data, val_data = split["train"], split["test"]
        epoch = 5
        if test_run:
            train_data = train_data.select(range(256))
            val_data = val_data.select(range(64))
            epoch = 10
        self._execute_pipeline(
            train_data=train_data,
            test_data=val_data,
            n_heads=12,
            dim=768,
            n_layers=12,
            lr=2e-5,
            epochs=epoch,
            batch_size=8,
            save_model=True,
            accumulation_steps=4,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            model_save_dir=model_save_dir,
            save_every_n_steps=10_000_000,
            upload_models=True,
            instruct=True,
        )

    def main(
        self,
    ):
        docs_to_take = 3_500_000  # 3_500_000
        save_every_n_steps = 72_917  # 109_375
        test_docs_to_take = 5000
        print("Loading Training Data")
        train_downloader = DataDownload(
            dataset_name="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            split="train",
            streaming=True,
            docs_to_take=docs_to_take,
        )
        train_stream = train_downloader()
        print("Loaded Training Data")
        print("Loading Val Data")
        val_downloader = DataDownload(
            dataset_name="Salesforce/wikitext",
            config_name="wikitext-2-raw-v1",
            split="validation",
            streaming=False,
        )
        val_data = val_downloader()
        val_data = val_data.select(range(min(test_docs_to_take, len(val_data))))
        print("Loaded val Data")

        print("Starting Training Pipeline...")
        self._execute_pipeline(
            train_data=train_stream,
            test_data=val_data,
            n_heads=12,
            dim=768,
            n_layers=12,
            lr=5e-4,
            epochs=1,
            batch_size=12,
            save_model=True,
            accumulation_steps=16,
            save_every_n_steps=save_every_n_steps,
            docs_to_take=docs_to_take,
        )

    def continued_pretrain(
        self,
        checkpoint_path: str = "resources/SmoLLM/run_2026_06_27__16_32/pytorch_model.bin",
        tokenizer_path: str = "resources/SmoLLM/run_2026_06_27__16_32/tokenizer.json",
        model_save_dir: str = "resources/SmoLLM-100M-EOS-continued",
    ):
        docs_to_take = 96  # 250_000
        save_every_n_steps = 24  # 5000
        test_docs_to_take = 50  # 5000
        print("Loading Training Data")
        train_downloader = DataDownload(
            dataset_name="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            split="train",
            streaming=True,
            docs_to_take=docs_to_take,
        )
        train_stream = train_downloader()
        print("Loaded Training Data")
        print("Loading Val Data")
        val_downloader = DataDownload(
            dataset_name="Salesforce/wikitext",
            config_name="wikitext-2-raw-v1",
            split="validation",
            streaming=False,
        )
        val_data = val_downloader()
        val_data = val_data.select(range(min(test_docs_to_take, len(val_data))))
        print("Loaded val Data")

        print("Starting Training Pipeline...")
        self._execute_pipeline(
            train_data=train_stream,
            test_data=val_data,
            n_heads=12,
            dim=768,
            n_layers=12,
            lr=1e-4,
            epochs=1,
            batch_size=12,
            save_model=True,
            accumulation_steps=16,
            docs_to_take=docs_to_take,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            model_save_dir=model_save_dir,
            save_every_n_steps=save_every_n_steps,
            upload_models=True,
        )

    def micro_train_loop(
        self,
    ):
        docs_to_take = 128
        save_every_n_steps = 16
        test_docs_to_take = 10

        print("Loading Training Data")
        train_downloader = DataDownload(
            dataset_name="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            split="train",
            streaming=True,
            docs_to_take=docs_to_take,
        )
        train_stream = train_downloader()
        print("Loaded Training Data")
        print("Loading Val Data")
        val_downloader = DataDownload(
            dataset_name="Salesforce/wikitext",
            config_name="wikitext-2-raw-v1",
            split="validation",
            streaming=False,
        )
        val_data = val_downloader()
        val_data = val_data.select(range(min(test_docs_to_take, len(val_data))))
        print("Loaded val Data")

        print("Starting Training Pipeline...")
        self._execute_pipeline(
            train_data=train_stream,
            test_data=val_data,
            n_heads=12,
            dim=768,
            n_layers=12,
            lr=5e-4,
            epochs=2,
            batch_size=12,
            save_model=True,
            accumulation_steps=16,
            save_every_n_steps=save_every_n_steps,
            docs_to_take=docs_to_take,
        )

    def _print_model_size(
        self,
        model,
    ):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    def train_tokenizer(
        self,
        sample_size: int = 500_000,
    ):
        downloader = DataDownload()
        hf_data = downloader()
        tokenizer = SmoLLMTokenizer()
        tokenizer.train(hf_dataset=hf_data, num_docs_for_training=sample_size)


if __name__ == "__main__":
    runner = SmoLLMRunner()
    runner.instruct_tune(
        model_save_dir="resources/SmoLLM-100M-Instruct_test",
        checkpoint_path="resources/SmoLLM/eos/run_2026_07_04__19_35/checkpoint-epoch-1/pytorch_model.bin",
        tokenizer_path="resources/SmoLLM/eos/run_2026_07_04__19_35/checkpoint-epoch-1/tokenizer.json",
        test_run=True,
    )
