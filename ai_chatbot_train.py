import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

tf.keras.backend.clear_session()
print("Tensorflow Version: ", tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class train_dataset:
    def __init__(self, dataset):
        with open(dataset) as file:
            self.dataset = json.load(file)

        self.training_y = []
        self.training_x = []
        self.labels = []
        self.responses = []

        for intent in self.dataset['intents']:
            for pattern in intent['patterns']:
                self.training_x.append(pattern)
                self.training_y.append(intent['tag'])
            self.responses.append(intent['responses'])

            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])

        self.NUM_CLASSES = len(self.labels)


        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.training_y)
        self.training_y = self.label_encoder.transform(self.training_y)


        VOCAB_SIZE = 200
        EMBEDDING_DIM = 8
        MAX_LENGTH = 100
        TRUNC_TYPE = 'post'
        OOV_TOK = "<OOV>"

        self.tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
        self.tokenizer.fit_on_texts(self.training_x)
        sequences = self.tokenizer.texts_to_sequences(self.training_x)
        self.padded = pad_sequences(sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.NUM_CLASSES, activation='softmax')
        ])

        self.model.summary()


        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(self.padded, np.array(self.training_y), epochs=500, batch_size=32, verbose=1)
        self.model.save('ai_model')


        def plot_graphs(history, string):
            plt.plot(history.history[string])
            # plt.plot(history.history['val_'+string])
            plt.xlabel("Epochs")
            plt.ylabel(string)
            # plt.legend([string, 'val_'+string])
            plt.show()


        plot_graphs(history, 'accuracy')
        plot_graphs(history, 'loss')

        # to save the fitted tokenizer
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # to save the fitted label encoder
        with open('label_encoder.pickle', 'wb') as ecn_file:
            pickle.dump(self.label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

new = train_dataset('intents.json')