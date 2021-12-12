from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
import pickle

app = Flask(__name__)

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

model = pickle.load(open("model.pkl",'rb'))
vectorizer = pickle.load(open("vectorizer.pkl",'rb'))

def transform_text(text):
    text = text.lower()     # Converting to lower case

    word_arr = nltk.word_tokenize(text)     # Tokenizing
    correct = []

    for word in word_arr:
        if (word.isalnum()) and (word not in STOPWORDS) and (word not in punctuation):      # Removal of special char, stop words, punctuation
            correct.append(ps.stem(word))                                                   # Stemming

    ans = " ".join(correct)
    return ans

@app.route('/', methods=["GET", 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        prediction = "Testing Going On"

    title = request.form["text"]                            # User input
    transformed_text = transform_text(title)                # Transformed Text
    vector = vectorizer.transform([transformed_text])       # Vectorize the text
    output = model.predict(vector)                          # Predict output

    if output == 0:
        return render_template('index.html', prediction_text="Business News")
    elif output == 1:
        return render_template('index.html', prediction_text="Entertainment News")
    elif output == 2:
        return render_template('index.html', prediction_text="Science News")
    elif output == 3:
        return render_template('index.html', prediction_text="Sports News")

if __name__ == "__main__":
    app.run()
