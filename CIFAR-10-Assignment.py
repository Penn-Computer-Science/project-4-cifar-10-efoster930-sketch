
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import seaborn as sns

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

batch_size = 32
num_classes = 10
epochs = 8

model = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Conv2D(64,(5,5),padding = 'same',activation = 'relu',input_shape=input_shape), #What it's basically doing is looking for edges in the image.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(), #Downsampling the image to reduce the number of parameters and computation in the network.
        #tf.keras.layers.Dropout(0,25), #Dropout to prevent overfitting.
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64,(5,5),padding = 'same',activation = 'relu',input_shape=input_shape), #What it's basically doing is looking for edges in the image.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(), #Downsampling the image to reduce the number of parameters and computation in the network.
        #tf.keras.layers.Dropout(0,25), #Dropout to prevent overfitting.
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(), #Flattening the 2D arrays for fully connected layers
        tf.keras.layers.Dense(num_classes, activation = 'softmax') #Output layer with softmax activation for multi-class classification. A softmax function is used to convert logits to probabilities. Logits are the raw, unnormalized scores outputted by the last layer of the neural network.
])



#We use categorical crossentropy as our loss function because we have more than two classes.
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,epochs=epochs,validation_split=(0.1))

fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history['loss'], color='b', label="Training Loss")  # Training loss in blue
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")  # Validation loss in red
# Add a legend to the first subplot for clarity
legend = ax[0].legend(loc='best', shadow=True)
# Add labels and title to the first subplot
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

# Plot the training and validation accuracy on the second subplot
ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")  # Training accuracy in blue
ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")  # Validation accuracy in red
# Add a legend to the second subplot for clarity
legend = ax[1].legend(loc='best', shadow=True)
# Add labels and title to the second subplot
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

# Display the plots
plt.tight_layout()  # Adjust layout to prevent overlap between subplots
plt.show()  # Show the figure







#generate the confusion matrix
# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis=1) 
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# Visualize the activations of each layer for a sample image
sample_image = x_test[0]  # Select a sample image from the test set
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension

# Create a model that outputs the activations of each layer
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)

# Get the activations for the sample image
activations = activation_model.predict(sample_image)

# Plot the activations for each layer
for layer_index, activation in enumerate(activations):
  num_filters = activation.shape[-1]  # Number of filters in the layer
  size = activation.shape[1]  # Size of the feature map

  # Create a grid to display the activations
  grid_size = int(np.ceil(np.sqrt(num_filters)))
  fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
  fig.suptitle(f'Layer {layer_index + 1} Activations', fontsize=16)

  for i in range(grid_size * grid_size):
    ax = axes[i // grid_size, i % grid_size]
    if i < num_filters:
      ax.imshow(activation[0, :, :, i], cmap='viridis')
    ax.axis('off')

  plt.tight_layout()
  plt.show()

"""""""""""""""
Some differences that I can see between CIFAR-10 and MNIST is that CIFAR-10 takes more filters than MNIST, which means there's more complex images. It, of course, 
takes more epochs to process the images in CIFAR-10 than MNIST. Another thing that I saw was that it's hard to get a really good accuracy than in MNIST (I mean, in val_acc). 
A change that I had to make is just add an augmentation so that CIFAR doesn't see the same image over and over again. I added some BatchNormalization that helps with 
higher accuracy and make CIFAR train faster. I removed the Dropout to get more accuracy and added more Conv2D to look more details. Some improvements that I thought about was maybe make an 
improved version of my augmentation.

"""""""""""""""