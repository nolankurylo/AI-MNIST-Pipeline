import pandas as pd
import boto3
import io
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]


class DataAcquisition:

    def __init__(self):
        pass

    @staticmethod
    def acquire_data():
        """ Retrieve train and test dataframes currently stored on AWS S3
        :return: training, testing data and labels
        """
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

        train_obj = s3.get_object(Bucket='ai-mnist-pipeline-bucket', Key='fashion-mnist_train.csv')
        train_data = pd.read_csv(io.BytesIO(train_obj['Body'].read()))
        train_y = train_data[['label']]
        train_X = train_data.drop('label', axis=1)

        test_obj = s3.get_object(Bucket='ai-mnist-pipeline-bucket', Key='fashion-mnist_test.csv')
        test_data = pd.read_csv(io.BytesIO(test_obj['Body'].read()))
        test_y = test_data[['label']]
        test_X = test_data.drop('label', axis=1)

        train_X.to_csv("data/train_X.csv", index=False)
        train_y.to_csv("data/train_y.csv", index=False)
        test_X.to_csv("data/test_X.csv", index=False)
        test_y.to_csv("data/test_y.csv", index=False)

        return train_X, test_X, train_y, test_y







