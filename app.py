from flask import Flask, request, render_template
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load artifacts
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    if request.method == 'POST':
        text = request.form['text']
        if not text.strip():
            return render_template('index.html', prediction="Please enter text.")
        clean_text = preprocess_text(text)
        vec_text = vectorizer.transform([clean_text])
        pred = model.predict(vec_text)[0]
        prob = model.predict_proba(vec_text)[0][1] * 100  # Spam probability
        prediction = 'Spam' if pred == 1 else 'Not Spam'
        probability = f"{prob:.2f}%"
    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)