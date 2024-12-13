import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.models import Sequential
import pickle

train = pd.read_csv("train.csv", encoding="iso-8859-1")
test = pd.read_csv("test.csv", encoding="iso-8859-1")


train = train.dropna()
test = test.dropna()



x_train = train["text"].str.lower().to_list()
y_train = train["sentiment"].replace({'positive': 1, 'negative': 0, 'neutral': 2})
x_test = test["text"].str.lower().to_list()
y_test = test["sentiment"].replace({'positive': 1, 'negative': 0, 'neutral': 2})


tokenize = Tokenizer(num_words=6000)
tokenize.fit_on_texts(x_train)


x_train_dataset = tokenize.texts_to_sequences(x_train)
X_train = pad_sequences(x_train_dataset, maxlen=200)

x_test_dataset = tokenize.texts_to_sequences(x_test)
X_test = pad_sequences(x_test_dataset, maxlen=200)

model = Sequential([
    keras.layers.Embedding(input_dim=6000, output_dim=128),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(64),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax') 
])


model.compile(loss=keras.losses.sparse_categorical_crossentropy, 
              optimizer=keras.optimizers.Adam(), 
              metrics=["accuracy"])


model.fit(X_train, y_train, batch_size=32, epochs=10)


model.save("sentiment.h5")


with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenize, f)
