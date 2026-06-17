from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv

load_dotenv()


class DataDownload:
    def __init__(
        self,
        dataset_name: str = "BabyLM-community/BabyLM-2026-Strict-Small",
    ):
        self.dataset = self._load_dataset(dataset_name=dataset_name)
        pass

    def _load_dataset(
        self,
        dataset_name: str,
    ):
        return load_dataset(dataset_name)

    def __call__(
        self,
    ):
        return self.dataset


if __name__ == "__main__":
    dd = DataDownload()

    my_dataset = dd()

    print("Dataset Structure:\n", my_dataset)

    print(my_dataset["train"][0])
