import numpy as np
from keras.layers import Embedding, Dense
from keras.callbacks import ModelCheckpoint, Callback
from keras_preprocessing.text import Tokenizer
from keras.models import Model
import pandas as pd

path = "IMDB Dataset.csv"
csv = pd.read_csv(path)

Reviews = csv['review']
Sentiments = csv['sentiment']

def tokenize(text, tokenizer):
    tokenizer.fit_on_texts(text)
    new = []
    for x in text:
        x = x.replace('<br />', '')
        new.append(x)
    sequences = tokenizer.texts_to_sequences(new)
    vocab_size = len(tokenizer.word_index) + 1
    return sequences, vocab_size

tokenizer = Tokenizer()
context, vocab_size = tokenize(Reviews, tokenizer)

target = []
for i in Sentiments:
    if i == 'positive':
        target.append(1)
    else:
        target.append(0)

temp = []
for list in context:
    tokens = []
    for token in list:
        tokens.append(token)
    temp.append(token)
context = temp

num_classes = 1

class SentimentAnalysis(Model):
    def __init__(self, vocab_size, dimensions, num_classes):
        super().__init__()
        self.emb = Embedding(vocab_size, dimensions, name='embedding')
        self.dense = Dense(16)
        self.dense1 = Dense(num_classes)

    def __call__(self, inputs, training=True):
        x = inputs = self.emb(inputs)
        x = self.dense(x)
        x = self.dense1(x)
        return x

context = np.array(context)
target = np.array(target)

json = tokenizer.to_json()
with open("Tokenizer/Tokenizer.json", 'w') as f:
    f.write(json)

print(target)
print(context)

model = SentimentAnalysis(vocab_size, 4, num_classes)
callback = ModelCheckpoint("Models/ModelWeights.h5", monitor='accuracy', save_best_only=True, save_weights_only=True)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(context, target, epochs=25, callbacks=[callback])

emb_weights = model.get_layer(name='embedding').get_weights()[0]

class Word2Vec(Model):
    def __init__(self, vocab_size, dimensions, weights):
        super().__init__()
        self.emb = Embedding(vocab_size, dimensions, name='embedding', weights=weights)

    def __call__(self, inputs):
        x = inputs = self.emb(inputs)
        return x

Word2Vec = Word2Vec(vocab_size, 4, emb_weights)
Word2Vec.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
Word2Vec.save_weights("Models/Word2Vec.h5")


