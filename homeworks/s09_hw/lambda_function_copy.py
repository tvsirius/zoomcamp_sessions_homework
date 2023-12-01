#!/usr/bin/env python
# coding: utf-8

# import tflite_runtime.interpreter as tflite

import tensorflow.lite as tflite

from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np

interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


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
    x /= 127.5
    x -= 1.
    return x

classes = ['bee',]

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def predict(url):
    img = download_image(url)
    img = img = prepare_image(img, (150, 150))

    # x = np.array(img, dtype='float32')
    # X = np.array([x])
    # X = preprocess_input(X)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    X = test_datagen.flow(img_array, batch_size=1).next()

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


# print(lambda_handler({'url':'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'}, None))