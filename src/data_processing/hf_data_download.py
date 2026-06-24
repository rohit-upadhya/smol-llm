from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv

load_dotenv()


class DataDownload:
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        config_name: str = "sample-10BT",
        docs_to_take: int = 2_500_000,
        streaming: bool = True,
        split: str = "train",
    ):
        self.config_name = config_name
        self.docs_to_take = docs_to_take
        self.dataset_name = dataset_name
        self._load_dataset(streaming=streaming, split=split)

    def _load_dataset(
        self,
        streaming: bool = True,
        split: str = "train",
    ):
        dataset = load_dataset(
            self.dataset_name,
            name=self.config_name,
            split=split,
            streaming=streaming,
        )
        print(f"Number of documents loaded : {self.docs_to_take}.")
        if streaming and self.docs_to_take is not None:
            self.dataset = dataset.take(self.docs_to_take)
        else:
            self.dataset = dataset

    def __call__(
        self,
    ):
        return self.dataset


if __name__ == "__main__":
    dd = DataDownload(docs_to_take=10)
    my_dataset = dd()

    for i, doc in enumerate(my_dataset):
        print(f"\n--- Document {i+1} ---")
        print(doc.get("text", "") + "\n")
