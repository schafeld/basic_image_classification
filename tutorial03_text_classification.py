import tensorflow as tf
from tensorflow import keras
import numpy as np
# I had numpy 1.16.3, tutorial needed specific version:
# pip install numpy==1.16.1

print("TensorFlow version: ", tf.__version__)

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000) # use 88,000 words


word_index = data.get_word_index()

word_index = {key:(value + 3) for key, value in word_index.items()} # +3 to provide space for special keys in dictionary, see below
word_index["<PAD>"] = 0 # padding to equalize text lengths
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # reverse key value order

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)

## Debug
# print(len(train_data), len(test_data)) # check that training and testing data have same lenght
# print(len(test_data[0]), len(test_data[1]))


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

## Debugging output
# print(decode_review(test_data[1]))


## Model definition
## Saves model in the end, thus only needed once in the beginning
'''
# define layers
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16)) # embedding size needs to correspond with num_words above
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid")) # sigmoid -> 0...1

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) # binary_crossentropy -> 1 || 0

x_val = train_data[:10000] # take first 10,000 datasets from data for validation
x_train = train_data[10000:] # take remaining datasets from data for training

y_val = train_labels[:10000] # take first 10,000 datasets from labels for validation
y_train = train_labels[10000:] # take remaining datasets from labels for training

# fit model

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results) # My output diverts clearly from tutorial (very low accuracy and high loss in last iteration).

model.save("movie_model.h5")
'''

def review_encode(s):
    encoded = [1] # set starting tag as defined above

    for word in s:
        if word in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2) # if word not in word index list use UNK tag
    return encoded


model = keras.models.load_model("movie_model.h5")

with open("movie_review.txt", encoding="utf-8") as f:
    # with means no need to close file after use
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").replace("\n", " ").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        print("------------------------ \n\n")
