
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


print(tf.__version__)

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("Any NaN Training:",np.isnan(x_train).any())
print("Any NaN Testing:",np.isnan(x_test).any())

from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

input_shape = (32,32,3) 

batch_size = 128
num_classes = 10
epochs = 10


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(5,5),padding = 'same',activation = 'relu',input_shape=input_shape), #What it's basically doing is looking for edges in the image.
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.MaxPool2D(), #Downsampling the image to reduce the number of parameters and computation in the network.
        tf.keras.layers.Dropout(0,25), #Dropout to prevent overfitting.
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.Flatten(), #Flattening the 2D arrays for fully connected layers
        tf.keras.layers.Dense(num_classes, activation = 'softmax') #Output layer with softmax activation for multi-class classification. A softmax function is used to convert logits to probabilities. Logits are the raw, unnormalized scores outputted by the last layer of the neural network.
])


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,validation_split=(0.1))

