import base64
import json
import numpy as np
from keras.models import load_model
from PIL import Image
from io import BytesIO

model = load_model('/var/task/model.h5')


def lambda_handler(event, context):
    """ AWS Lambda entry point. Takes in an event with a base64 encoding of the image to be predicted on, encoding
    is converted into NumPy representation, preprocessed, dimensionally reduced and the prediction is made & returned.
    :param event: contains base64 encoding of image
    :param context: required for AWS Lambda, not in use
    :return: http status, CORS headers, and AI label prediction
    """
    # Convert base64 to numpy representation of image, resizing to 28x28 pixels
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    image = image.resize((28, 28))

    # Preprocess and dimensionally reduce the image
    np_img = np.array(image) / 255.0

    # Apply AI to make prediction
    probabilities = model(np_img.reshape(-1, 28, 28, 1))
    label = np.argmax(probabilities)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'

        },
        'body': json.dumps(
            {
                "predicted_label": int(label),
            }
        )
    }
