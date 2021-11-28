import base64
import json
import numpy as np
from keras.models import load_model
from PIL import Image
from io import BytesIO


model = load_model('/var/task/model.h5')


def lambda_handler(event, context):
    image_bytes = event['body'].encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    print("3")
    image = image.resize((28, 28))
    np_img = np.array(image) / 255.0

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
