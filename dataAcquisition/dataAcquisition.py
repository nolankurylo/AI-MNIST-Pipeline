import pandas as pd
import boto3
import io
import os

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
        y_train = train_data[['label']]
        X_train = train_data.drop('label', axis=1)

        test_obj = s3.get_object(Bucket='ai-mnist-pipeline-bucket', Key='fashion-mnist_test.csv')
        test_data = pd.read_csv(io.BytesIO(test_obj['Body'].read()))
        y_test = test_data[['label']]
        X_test = test_data.drop('label', axis=1)

        X_train.to_csv("data/X_train.csv", index=False)
        y_train.to_csv("data/y_train.csv", index=False)
        X_test.to_csv("data/X_test.csv", index=False)
        y_test.to_csv("data/y_test.csv", index=False)

        return X_train, X_test, y_train, y_test







