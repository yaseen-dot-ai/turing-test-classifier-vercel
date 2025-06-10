import time
import json
import os
import boto3

class SyncS3Client:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        self.bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET")

    def get_object(self, key: str) -> str:
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def put_object(self, key: str, body: str):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body)

    def upload_parquet(self, key: str, df):
        import io
        out_buffer = io.BytesIO()
        df.to_parquet(out_buffer, index=False)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=out_buffer.getvalue())

s3_client = SyncS3Client()

def utc_stamp(ms=False):
    t = time.time()
    return str(int(t * 1000)) if ms else str(int(t))

def s3_put(path, data):
    if s3_client.bucket:
        s3_client.put_object(path, data)
    else:
        print(f"[S3 PUT] {path}")

def save_parquet_to_s3(path, df):
    if s3_client.bucket:
        s3_client.upload_parquet(path, df)
    else:
        print(f"[S3 PARQUET] {path}")

def write_local(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f) 