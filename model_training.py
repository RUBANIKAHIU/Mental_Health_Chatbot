import json
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# Load data from JSON file
with open("D:\FINAL P\data.json") as file:
    data = json.load(file)

questions = []
answers = []
labels = []

# Extract questions, answers, and labels
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        questions.append(pattern)
        answers.append(random.choice(intent["responses"]))
        labels.append(intent["tag"])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    questions, labels, test_size=0.2, random_state=42
)

# Convert y_train and y_test to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# One-hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
train_y = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
test_y = enc.transform(y_test.reshape(-1, 1)).toarray()

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")

max_len = 128

x_train_encodings = tokenizer(
    x_train,
    add_special_tokens=True,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True,
)

x_test_encodings = tokenizer(
    x_test,
    add_special_tokens=True,
    max_length=max_len,
    truncation=True,
    padding="max_length",
    return_tensors="tf",
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True,
)

# Build model
input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids, attention_mask=input_mask)[0]
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = tf.keras.layers.Dense(128, activation="relu")(out)
out = tf.keras.layers.Dropout(0.5)(out)
out = tf.keras.layers.Dense(64, activation="relu")(out)
y = tf.keras.layers.Dense(len(enc.categories_[0]), activation="softmax")(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=5e-5, epsilon=1e-08, decay=0.01, clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

#Train model
x_train_input = {"input_ids": x_train_encodings["input_ids"], "attention_mask": x_train_encodings["attention_mask"]}
train_history = model.fit(
x=x_train_input,
y=train_y,
validation_split=0.1,
shuffle=True,
epochs=150,
batch_size=16,
callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

#Evaluate model on test set
from sklearn.metrics import precision_score

x_test_input = {"input_ids": x_test_encodings["input_ids"], "attention_mask": x_test_encodings["attention_mask"]}

#Predict labels for test set
y_pred = np.argmax(model.predict(x_test_input), axis=-1)
y_true = np.argmax(test_y, axis=1)

#Calculate precision score=1)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
test_loss, test_acc = model.evaluate(x=x_test_input, y=test_y)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

#Classification report and confusion matrix
y_pred = model.predict(x_test_input)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(test_y, axis=1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

model.save('my_model.h5')

