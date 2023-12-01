import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Define classes
classes = ['bee']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x.astype('float32')
    x /= 255.0  # Rescale to the range [0.0, 1.0]
    return x


def predict(url):
    img = download_image(url)
    img = prepare_image(img, (150, 150))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Preprocess the input image
    x = preprocess_input(img_array)

    # Set input tensor
    interpreter.set_tensor(input_index, x)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    preds = interpreter.get_tensor(output_index)

    # Convert predictions to a dictionary
    float_predictions = preds[0].tolist()
    result = dict(zip(classes, float_predictions))
    print(result)

    return result


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


# Example usage
#print(lambda_handler({'url': 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'}, None))
