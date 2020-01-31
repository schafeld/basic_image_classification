import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

# Split data into training and testing data
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Labels are just numbers, here are the corrsponding names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalize pixel brightness values
train_images = train_images/255.0
test_images = test_images/255.0

# Debugging output
# print(train_labels[0]) # print single label number
# print(train_images[7]) # image pixel data as numbers array of arrays

# Show image
# plt.imshow(train_images[7]) # in false colors, greenish
plt.imshow(train_images[7], cmap=plt.cm.binary) # grayscale colors
plt.colorbar() # optional
plt.grid(False) # optional, grid overlay
plt.show()
