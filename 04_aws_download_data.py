import boto3

s3 = boto3.client('s3')
bucket_name = 'thesisroadclassification'
object_key = 'train.zip'
local_file_path = 'train.zip'

s3.download_file(bucket_name, object_key, local_file_path)

!unzip 'train.zip' -d 'data'