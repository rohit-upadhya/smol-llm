import os
import boto3
from urllib.parse import urlparse
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()


class S3Manager:
    def __init__(self, bucket_name: str):
        self.s3_client = boto3.client("s3")
        self.bucket_name = bucket_name

    def upload_folder(self, local_folder: str, s3_prefix: str):
        if not os.path.exists(local_folder):
            print(f"Directory not found: {local_folder}")
            return

        print(f"Starting upload to s3://{self.bucket_name}/{s3_prefix}/...")

        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)

                relative_path = os.path.relpath(local_path, local_folder)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                print(f"Uploading: {relative_path}")
                try:
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
                except NoCredentialsError:
                    print("ERROR: AWS credentials not found.")
                    return

        print("Upload completed successfully!")

    def download_from_uri(self, s3_uri: str, local_folder: str):
        parsed = urlparse(s3_uri)
        # Using the bucket from the URI in case it differs from the default,
        # but it will typically match self.bucket_name
        target_bucket = parsed.netloc
        s3_prefix = parsed.path.lstrip("/")

        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        print(f"Starting download from {s3_uri} to {local_folder}/...")

        paginator = self.s3_client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=target_bucket, Prefix=s3_prefix):
                if "Contents" not in page:
                    print(f"No files found at {s3_uri}.")
                    continue

                for obj in page["Contents"]:
                    s3_key = obj["Key"]

                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_path = os.path.join(local_folder, relative_path)

                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    print(f"Downloading: {relative_path}")
                    self.s3_client.download_file(target_bucket, s3_key, local_path)

        except NoCredentialsError:
            print("ERROR: AWS credentials not found.")
            return

        print("Download completed successfully!")


if __name__ == "__main__":
    s3 = S3Manager(bucket_name="smol-lm-bucket")

    # s3.upload_folder(
    #     local_folder="resources/SmoLLM-100M-Baby-LM-Base",
    #     s3_prefix="model_weights"
    # )

    s3.download_from_uri(
        s3_uri="s3://smol-lm-bucket/model_weights/run_2026_06_28__13_06/checkpoint-epoch-1/",
        local_folder="resources/SmoLLM/run_2026_06_27__16_32",
    )
