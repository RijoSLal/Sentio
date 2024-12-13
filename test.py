# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle


# model = load_model('sentiment.h5')


# with open('tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

# text = ["the service were amazing"]


# text_seq = tokenizer.texts_to_sequences(text)
# text_padded = pad_sequences(text_seq, maxlen=200)


# prediction = model.predict(text_padded)


# print("Prediction shape:", prediction.shape)
# print("Prediction values:", prediction)
string="hello123 bro"
print(string.lower())