# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import pickle

# # Load the pre-trained model
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model.pkl)

# # Load the TF-IDF vectorizer
# with open('vectorizer.pkl', 'rb') as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         text = request.form['text']
#         # Vectorize the input text
#         text_vectorized = vectorizer.transform([text])
#         # Make prediction
#         prediction = model.predict(text_vectorized)
#         # Convert prediction to human-readable format
#         if prediction[0] == 1:
#             result = 'Real News'
#         else:
#             result = 'Fake News'
#         return render_template('result.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle



app = Flask(__name__)

# Specify the path to the model file
model_path = 'model.pkl'



# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Vectorize the input text
        text_vectorized = vectorizer.transform([text])
        # Make prediction
        prediction = model.predict(text_vectorized)
        # Convert prediction to human-readable format
        if prediction[0] == 1:
            result = 'Real News'
        else:
            result = 'Fake News'
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

