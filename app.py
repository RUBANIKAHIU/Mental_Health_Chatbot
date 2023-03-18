from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
import numpy as np
import json



app = Flask(__name__)

from transformers import TFBertModel

# Register the TFBertModel as a custom object in Keras
tf.keras.utils.get_custom_objects()['TFBertModel'] = TFBertModel

# Load the trained model
model= tf.keras.models.load_model("D:\FINAL P\my_model.h5")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")

# Load the intents file
with open("D:\FINAL P\data.json") as file:
    intents = json.load(file)

# Define a function to generate a response to user input
def predict_intent(text):
    # Tokenize the input text
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True, padding='max_length')
    input_ids = np.array(input_ids)
    input_mask = np.where(input_ids != 0, 1, 0)
    input_ids = tf.expand_dims(input_ids, 0)
    input_mask = tf.expand_dims(input_mask, 0)

    # Generate the model's prediction
    outputs = model([input_ids, input_mask])
    prediction = tf.argmax(outputs, axis=-1)
    predicted_label = prediction.numpy()[0]
    
    # Look up the corresponding intent and response
    intent = intents['intents'][predicted_label]
    response = np.random.choice(intent['responses'])

    return response

# Define the routes for the web app
@app.route('/')
def home():
    return render_template('george.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    response = predict_intent(text)
    data={"answer": response, "text": text}
    
    return jsonify(data)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)