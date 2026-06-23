from src.data_processing.dataset import create_dateset
from src.models.tokenizer import SmoLLMTokenizer
from src.data_processing.hf_data_download import DataDownload
from src.models.models import SmoLLM
from src.training.training import SmoLLMTrainer


class SmoLLMRunner:
    def _execute_pipeline(
        self,
        data,
        test_size: float,
        n_heads: int,
        dim: int,
        n_layers: int,
        lr: float,
        epochs: int,
        batch_size: int = 4,
    ):
        split_data = data.train_test_split(test_size=test_size)

        tokenizer = SmoLLMTokenizer()
        tokenizer.load_or_train(hf_dataset=split_data["train"])
        pad_token_id = tokenizer.tokenizer.token_to_id("[PAD]")

        train_loader = create_dateset(
            dataset=split_data["train"],
            tokenizer=tokenizer,
            shuffle=True,
            batch_size=batch_size,
        )

        test_loader = create_dateset(
            dataset=split_data["test"],
            tokenizer=tokenizer,
            shuffle=False,
            batch_size=batch_size,
        )

        print(f"Number of Train Batches: {len(train_loader)}")
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
            ignore_index=pad_token_id,
        )

        trainer.fit(
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            epochs=epochs,
        )

    def main(
        self,
    ):
        downloader = DataDownload()
        hf_data = downloader()

        full_data = hf_data["train"]
        self._execute_pipeline(
            data=full_data,
            test_size=0.1,
            n_heads=12,
            dim=768,
            n_layers=8,
            lr=5e-4,
            epochs=3,
            batch_size=2,
        )

    def micro_train_loop(
        self,
    ):
        downloader = DataDownload()
        hf_data = downloader()

        micro_data = hf_data["train"].select(range(500))
        self._execute_pipeline(
            data=micro_data,
            test_size=0.1,
            n_heads=4,
            dim=128,
            n_layers=2,
            lr=1e-3,
            epochs=10,
            batch_size=2,
        )


if __name__ == "__main__":
    runner = SmoLLMRunner()
    runner.main()
