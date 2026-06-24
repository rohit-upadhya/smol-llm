from src.data_processing.dataset import create_dateset
from src.models.tokenizer import SmoLLMTokenizer
from src.data_processing.hf_data_download import DataDownload
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
        save_every_n_steps: int = 250_000_000,
    ):

        tokenizer = SmoLLMTokenizer()
        tokenizer.load_or_train(hf_dataset=train_data)
        pad_token_id = tokenizer.tokenizer.token_to_id("[PAD]")

        train_loader = create_dateset(
            dataset=train_data,
            tokenizer=tokenizer,
            shuffle=True,
            batch_size=batch_size,
        )

        test_loader = create_dateset(
            dataset=test_data,
            tokenizer=tokenizer,
            shuffle=False,
            batch_size=batch_size,
        )

        print(f"Number of Test Batches: {len(test_loader)}")

        model = SmoLLM(
            vocab_size=tokenizer.vocab_size,
            n_heads=n_heads,
            dim=dim,
            n_layers=n_layers,
        )

        trainer = SmoLLMTrainer(
            model=model,
            lr=lr,
            ignore_index=-100,
            save_model=save_model,
            accumulation_steps=accumulation_steps,
            save_every_n_steps=save_every_n_steps,
        )

        trainer.fit(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            epochs=epochs,
        )

    def main(
        self,
    ):
        print("Loading Training Data")
        train_downloader = DataDownload(
            dataset_name="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            split="train",
            streaming=True,
            docs_to_take=3_000_000,
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
        val_data = val_data.select(range(min(5000, len(val_data))))
        print("Loaded val Data")

        print("Starting Training Pipeline...")
        self._execute_pipeline(
            train_data=train_stream,
            test_data=val_data,
            n_heads=12,
            dim=768,
            n_layers=8,
            lr=5e-4,
            epochs=1,
            batch_size=4,
            save_model=True,
            accumulation_steps=4,
            save_every_n_steps=250_000_000,
        )

    def micro_train_loop(
        self,
    ):
        print("Loading Training Data")
        train_downloader = DataDownload(
            dataset_name="HuggingFaceFW/fineweb-edu",
            config_name="sample-10BT",
            split="train",
            streaming=True,
            docs_to_take=500,  # The micro-batch cap for local testing
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
        val_data = val_data.select(range(min(100, len(val_data))))
        print("Loaded val Data")

        print("Starting Training Pipeline...")
        self._execute_pipeline(
            train_data=train_stream,
            test_data=val_data,
            n_heads=1,
            dim=8,
            n_layers=1,
            lr=1e-3,
            epochs=2,
            batch_size=4,
            save_model=True,
            accumulation_steps=1,
            save_every_n_steps=50,
        )

    def train_tokenizer(
        self,
        sample_size: int = 100000,
    ):
        downloader = DataDownload()
        hf_data = downloader()
        tokenizer = SmoLLMTokenizer()
        tokenizer.train(hf_dataset=hf_data, num_docs_for_training=sample_size)


if __name__ == "__main__":
    runner = SmoLLMRunner()
    runner.micro_train_loop()
