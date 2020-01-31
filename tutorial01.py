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

def showImage():
    # Show image
    # plt.imshow(train_images[7]) # in false colors, greenish
    plt.imshow(train_images[7], cmap=plt.cm.binary) # grayscale colors
    plt.colorbar() # optional
    plt.grid(False) # optional, grid overlay
    plt.show()


model = keras.Sequential([ # sequence of layers definition
    keras.layers.Flatten(input_shape=(28,28)), # input mnist images are 28*28 pixels
    keras.layers.Dense(128, activation="relu"), # 128 neurons, rectified linear unit/relu, first hidden layer
    keras.layers.Dense(10, activation="softmax") # 10 output neurons due to 10 class names/labels, softmax distributes values from 0 to 1, i.e. individual label probability.
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5) # trigger actual training

# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# print("Tested accuracy: ", test_accuracy)

prediction = model.predict(test_images) # input is/ has to be a list
# print(class_names[np.argmax(prediction[1])])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual label: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
