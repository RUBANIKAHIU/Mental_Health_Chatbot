from flask import Flask, render_template, request
import random
import pickle
import json
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('D:\Folder neural\data.json').read())

words = pickle.load(open('D:\Folder neural\words.pkl', 'rb'))
classes = pickle.load(open('D:\Folder neural\classes.pkl', 'rb'))
model = tf.keras.models.load_model('D:\Folder neural\chatbot_model.h5')

@app.route('/')
def index():
    return render_template('george.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    message = request.form['message']
    message_words = nltk.word_tokenize(message)
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words]

    bag = [0] * len(words)
    for word in message_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    # Use the trained model to get a prediction for the user's message
    prediction = model.predict(np.array([bag]))[0]
    # Get the index of the predicted class
    predicted_class_index = np.argmax(prediction)
    # Get the corresponding class label
    predicted_class = classes[predicted_class_index]

    # Check if the predicted class has a corresponding response
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            break

    # If the predicted class doesn't have a corresponding response, return a default message
    else:
        response = "I'm sorry, I don't understand, but will have the response soon"

    return response

if __name__ == '__main__':
    app.run(debug=True)
