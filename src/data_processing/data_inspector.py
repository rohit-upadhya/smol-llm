import json
import os
import statistics
from src.models.tokenizer import SmoLLMTokenizer
from src.data_processing.hf_data_download import DataDownload


class DatasetInspector:
    def __init__(self, sample_size: int = 5000, target_tokens: int = 3_000_000_000):
        self.sample_size = sample_size
        self.target_tokens = target_tokens

        print("Initializing Tokenizer...")
        self.tokenizer = SmoLLMTokenizer()
        self.downloader = DataDownload(docs_to_take=self.sample_size)

    def analyze(self):
        print(f"\nStarting token analysis on {self.sample_size} documents...")
        dataset_stream = self.downloader()

        total_tokens = 0
        docs_processed = 0

        token_counts = []

        for item in dataset_stream:
            text = item.get("text", "")

            token_ids = self.tokenizer.encode(text)
            token_count = len(token_ids)

            total_tokens += token_count
            docs_processed += 1

            token_counts.append(token_count)

            if docs_processed % 1000 == 0:
                print(
                    f"Analyzed {docs_processed}/{self.sample_size} documents... ({total_tokens:,} tokens)"
                )

        avg_tokens_per_doc = total_tokens / docs_processed
        docs_needed = int(self.target_tokens / avg_tokens_per_doc)

        self._save_report(
            docs_processed, total_tokens, avg_tokens_per_doc, docs_needed, token_counts
        )

        del dataset_stream

    def _save_report(
        self,
        docs_processed,
        total_tokens,
        avg_tokens_per_doc,
        docs_needed,
        token_counts,
    ):
        insights = {
            "metadata": {
                "dataset": self.downloader.dataset_name,
                "documents_analyzed": docs_processed,
            },
            "averages": {
                "total_tokens_in_sample": total_tokens,
                "average_tokens_per_document": round(avg_tokens_per_doc, 2),
            },
            "distribution_stats": {
                "min_tokens_in_a_doc": min(token_counts),
                "max_tokens_in_a_doc": max(token_counts),
                "median_tokens_per_doc": statistics.median(token_counts),
            },
            "cloud_training_target": {
                "target_tokens": self.target_tokens,
                "estimated_documents_needed": docs_needed,
            },
            "raw_token_counts": token_counts,
        }

        os.makedirs("resources", exist_ok=True)
        output_file = "resources/fineweb_insights.json"

        with open(output_file, "w") as f:
            json.dump(insights, f, indent=4)

        print("\n" + "=" * 50)
        print("                 INSPECTION COMPLETE")
        print("=" * 50)
        display_insights = insights.copy()
        del display_insights["raw_token_counts"]
        print(json.dumps(display_insights, indent=4))
        print("=" * 50)
        print(f"\n🚀 TO HIT {self.target_tokens:,} TOKENS:")
        print(f"Update main.py to pass docs_to_take={docs_needed:,} to DataDownload!")


if __name__ == "__main__":
    inspector = DatasetInspector(sample_size=5000, target_tokens=3_000_000_000)
    inspector.analyze()
