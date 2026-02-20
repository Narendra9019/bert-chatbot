from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import json
import numpy as np

# Load intents
with open('intents.json') as file:
    data = json.load(file)

texts = []
labels = []
tags = []

# Prepare data
for intent in data['intents']:
    tags.append(intent['tag'])

for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(tags.index(intent['tag']))

labels = np.array(labels)  # ✅ IMPORTANT

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    return_tensors="tf"  # ✅ VERY IMPORTANT
)

# Load model
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(tags)
)

# ✅ IMPORTANT — use proper loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy']
)

# ✅ Proper dataset
dataset = tf.data.Dataset.from_tensor_slices((
    dict(encodings),
    labels
)).shuffle(100).batch(4)

# Train
model.fit(dataset, epochs=3)

# Save
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("✅ Training completed!")