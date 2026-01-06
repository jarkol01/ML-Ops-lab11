import boto3
import os

from settings import Settings

def download_s3_folder(bucket_name, local_dir):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    objects = list(bucket.objects.filter(Prefix=''))
    for obj in objects:
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key.endswith('/'):
            continue

        bucket.download_file(obj.key, target)
    print("Done!")
    

if __name__ == "__main__":
    settings = Settings()
    if not os.path.exists(settings.local_dir):
        os.makedirs(settings.local_dir)

    download_s3_folder(settings.bucket_name, settings.local_dir)