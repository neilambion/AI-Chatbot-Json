import tensorflow as tf
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

import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import pickle

with open('intents.json') as file:
    dataset = json.load(file)

def chat():
    model = tf.keras.models.load_model('ai_model')
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
        
    MAX_LEN = 100
    
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        result = model.predict(tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=MAX_LEN))
        result_index = np.argmax(result)
        if result[0][result_index] >= 0.9:
            tag = lbl_encoder.inverse_transform([result_index])

            for i in dataset['intents']:
                if i['tag'] == tag:
                    print("Chatbot: ", np.random.choice(i['responses']))
        else:
            print("Chatbot: I'm sorry I didn't understand. Please try again.")

chat()
