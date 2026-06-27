import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()


class S3Uploader:
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

        print("Upload completely successfully!")


if __name__ == "__main__":
    uploader = S3Uploader(bucket_name="smol-lm-bucket")
    uploader.upload_folder(
        local_folder="resources/SmoLLM-100M-Baby-LM-Base", s3_prefix="model_weights"
    )
