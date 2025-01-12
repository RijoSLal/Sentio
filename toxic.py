import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.models import Sequential
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import numpy as np


'''
there are some issue with the data, please verify data integrity before training model

'''



'''
this is better practice than other model and you can also preprocess using sklearn TfidfVectorizer for simple format
'''


def splitting_data(data):
    dataset=pd.read_csv(data)
    X=dataset["comment_text"]
    Y=dataset[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=24)
    return x_train,x_test,y_train,y_test


# labels -> comment_text,toxic,severe_toxic,obscene,threat,insult,identity_hate

def tokenize_model_compile_fit_save(x_train,x_test,y_train,y_test):
    tokenize = Tokenizer(num_words=6000,lower=True)
    tokenize.fit_on_texts(x_train)


    x_train_dataset = tokenize.texts_to_sequences(x_train)
    X_train = pad_sequences(x_train_dataset, maxlen=200)

    x_test_dataset = tokenize.texts_to_sequences(x_test)
    X_test = pad_sequences(x_test_dataset, maxlen=200)

    model = Sequential([
        keras.layers.Embedding(input_dim=6000, output_dim=128),
        keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(6,activation="sigmoid") 
    ])

    stop=EarlyStopping(monitor="loss",patience=2)

    model.compile(loss=keras.losses.BinaryCrossentropy(), 
                optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                metrics=["accuracy"])


    model.fit(X_train, y_train, batch_size=36, epochs=5,callbacks=[stop])
    print("model fitted successfully")

    loss,accuracy=model.evaluate(X_test,y_test)
    print(f"loss : {loss}\naccuracy : {accuracy}")

    model.save("sentiment_toxic.h5")
    print("model saved")


    with open('tokenizer_toxic.pkl', 'wb') as f:
        pickle.dump(tokenize, f)



def check_model_works():
    model = load_model('sentiment_toxic.h5')


    with open('tokenizer_toxic.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    text = ["toxic comments"]


    text_seq = tokenizer.texts_to_sequences(text)
    text_padded = pad_sequences(text_seq, maxlen=200)


    prediction = model.predict(text_padded)


    print("Prediction shape:", prediction.shape) #this should give 6
    print("Prediction values:", prediction)




