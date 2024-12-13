# Sentio - Sentiment Analysis Web App

**Sentio** is a sentiment analysis web application based on LSTM built using TensorFlow, Streamlit and other essential libraries. It allows users to classify text comments as **Positive**, **Negative**, or **Neutral** by simply entering text into a text field.

## Features

- Classifies comments into **Positive**, **Negative**, or **Neutral** categories.
- Provides a user-friendly interface using Streamlit.
- Uses a LSTM model for sentiment classification.
- Visualizes the sentiment prediction with color-coded progress bars.

## Tech Stack

- **Streamlit**: For building the web interface.
- **TensorFlow**: For deep learning and sentiment classification.
- **Keras**: For implementing the neural network model.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Pickle**: For saving and loading the tokenizer model.

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
```

## How to Run the App

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/sentio.git
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

## License

This project is licensed under the MIT License. Feel free to modify, use, and distribute the code in any way you like. See the LICENSE file for details.
