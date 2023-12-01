






#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://github.com/alexeygrigorev/large-datasets/releases/download/wasps-bees/bees-wasps.h5')


# In[2]:


import numpy as np

import tensorflow as tf
from tensorflow import keras

tf.__version__


# In[3]:


model = keras.models.load_model('bees-wasps.h5')


# In[4]:


model


# In[5]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)


# In[6]:


tflite_model = converter.convert()


# In[7]:


with open('bees-wasps.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[8]:


get_ipython().system('ls')


# In[9]:


import tensorflow.lite as tflite


# In[10]:


interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[11]:


interpreter.get_input_details()


# In[12]:


interpreter.get_output_details()


# In[13]:

# In[14]:


from io import BytesIO
from urllib import request

from PIL import Image

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


# In[15]:


img=download_image("https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg")


# In[16]:


img


# In[17]:


img=prepare_image(img, (150,150))


# In[ ]:





# In[18]:


img


# In[19]:


x = np.array(img, dtype='float32')
X = np.array([x])


# In[20]:


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# In[21]:


get_ipython().system('pip install keras-image-helper')


# In[22]:


get_ipython().system('pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime')


# In[23]:


import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


# In[24]:


img


# In[25]:


interpreter = tflite.Interpreter(model_path='bees-wasps.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[26]:


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# In[27]:


x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)


# In[29]:


X


# In[30]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[31]:


preds


# In[37]:


from tensorflow.keras.preprocessing.image import img_to_array


# In[33]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[34]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[38]:


img_array = img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

# Use the test_datagen to preprocess the image array
X = test_datagen.flow(img_array, batch_size=1).next()


# In[39]:


X


# In[40]:


interpreter.set_tensor(input_index, X)

# Invoke the interpreter
interpreter.invoke()

# Get the predictions
preds = interpreter.get_tensor(output_index)


# In[41]:


preds


# In[ ]:




