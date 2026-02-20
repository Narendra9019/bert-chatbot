from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import sqlite3
import random
import json

app = Flask(__name__)

model = TFBertForSequenceClassification.from_pretrained("model")
tokenizer = BertTokenizer.from_pretrained("model")

with open("intents.json") as file:
    intents = json.load(file)

tags = [intent['tag'] for intent in intents['intents']]

def get_response(text):
    inputs = tokenizer(text, return_tensors="tf")
    outputs = model(**inputs)
    prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
    tag = tags[prediction]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def chatbot_response():
    user_text = request.args.get('msg')
    response = get_response(user_text)

    # Store chat
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS chats (user TEXT, bot TEXT)")
    cursor.execute("INSERT INTO chats VALUES (?, ?)", (user_text, response))
    conn.commit()
    conn.close()

    return response

if __name__ == "__main__":
    app.run()
