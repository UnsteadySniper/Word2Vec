import numpy as np
from keras.layers import Embedding
from keras.models import Model
from keras_preprocessing.text import tokenizer_from_json

class Word2Vec(Model):
    def __init__(self, vocab_size, dimensions):
        super().__init__()
        self.emb = Embedding(vocab_size, dimensions, name='embedding')

    def __call__(self, inputs, training=True):
        x = inputs = self.emb(inputs)
        return x

with open("Tokenizer/Tokenizer.json", 'r') as f:
    json = f.read()
tokenizer = tokenizer_from_json(json)

vocab_size = len(tokenizer.word_index) + 1
dimensions = 4

Model = Word2Vec(vocab_size, dimensions)
Model.built = True
Model.load_weights('Models/Word2Vec.h5')

print("Type word and vector comes")

while True:
    word = input(" : ")
    sequence = tokenizer.texts_to_sequences([word])
    print(Model.predict(np.array(sequence)))
