import base64
import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

model = load_model('/var/task/basic_mnist_nn.h5')


def lambda_handler(event, context):
    print("helllooooo")
    print(event['body'])
    image_bytes = event['body'].encode('utf-8')
    print("2")
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    print("3")
    image = image.resize((28, 28))
    plt.imshow(image)

    probabilities = model(np.array(image).reshape(-1, 28, 28, 1))
    label = np.argmax(probabilities)

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": int(label),
            }
        )
    }
