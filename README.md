# Next Word Prediction using GRU

This repository implements a Next Word Prediction system using a GRU-based deep learning model. The project demonstrates how to preprocess text data, train a GRU neural network, and deploy an interactive web app for next-word prediction.

## Features

- **Dataset:** Uses Shakespeare's "Hamlet" from the NLTK Gutenberg corpus.
- **Preprocessing:** Text cleaning, tokenization, sequence creation, and padding.
- **Model:** Embedding layer, stacked GRU layers, and Dense output with softmax activation.
- **Training:** Early stopping to prevent overfitting, with train/test split.
- **Prediction:** Function to predict the next word given an input sentence.
- **Deployment:** Streamlit web app for real-time next-word prediction.
- **Persistence:** Saves trained model (`.h5`) and tokenizer (`.pkl`) for reuse.

## Getting Started

1. **Clone the repository**
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Run the notebook**  
   Open `PredictionUsingGRU.ipynb` to train and evaluate the model.
4. **Launch the web app**
   ```
   streamlit run app.py
   ```

## Usage

- Enter a sentence in the Streamlit app to get the predicted next word.
- The model can be retrained on different text by modifying the dataset.

## Requirements

- Python 3.7+
- TensorFlow, Keras, NLTK, Streamlit, scikit-learn, numpy, pandas

## License

This project is for educational purposes.
