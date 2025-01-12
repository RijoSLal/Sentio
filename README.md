# Sentio - Sentiment & Toxicity Analysis Web App

**Sentio** is a sentiment analysis web application based on LSTM built using TensorFlow, Streamlit and other essential libraries. It allows users to classify text comments as **Positive**, **Negative**, or **Neutral** and classify **Toxicity** contents in the comment by simply entering text into a text field.

## Features

- Classifies comments into **Positive**, **Negative**, or **Neutral** categories.
- Identify **Toxic** contents in comments
- Provides a user-friendly interface using Streamlit.
- Uses a LSTM model for sentiment classification.
- Visualizes the sentiment prediction and toxicity with color-coded progress bars.

## Tech Stack

- **Streamlit**: For building the web interface.
- **TensorFlow**: For deep learning and sentiment classification.
- **Keras**: For implementing the neural network model.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Pickle**: For saving and loading the tokenizer model.
- **sklearn**: For splitting data and preprocessing

## Requirements

To install the required packages, create a virtual environment and install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
streamlit
pandas
numpy
tensorflow-cpu
pickle-mixin
sklearn
```

## How to Run the App

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/RijoSLal/sentio.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run sentio.py
   ```

4. Open your browser and go to `http://localhost:8501` to use the Sentio app.

## Files

- **sentio.py**: The Streamlit app for the sentiment analysis interface.
- **sentiment.h5**: The trained sentiment analysis model.
- **tokenizer.pkl**: The tokenizer used to preprocess input text.
- **train.csv**: The training dataset.
- **test.csv**: The test dataset.
- **toxic_data.csv**: the dataset for toxicity detection
- **sentiment_toxic.h5**: The trained toxicity recognition model.
- **tokenizer_toxic.pkl**: The tokenizer used to preprocess toxic text.

## License

This project is licensed under the MIT License. Feel free to modify, use, and distribute the code in any way you like. See the LICENSE file for details.
