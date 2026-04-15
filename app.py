from flask import Flask, render_template, request
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    
    # Remove stopwords
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    
    return " ".join(words)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    score = None

    if request.method == 'POST':
        q1 = preprocess(request.form['q1'])
        q2 = preprocess(request.form['q2'])

        # TF-IDF with n-grams (better accuracy)
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
        vectors = vectorizer.fit_transform([q1, q2])

        # Cosine Similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        score = round(similarity, 2)

        # Adjusted threshold
        threshold = 0.5

        if similarity >= threshold:
            result = "Duplicate Question ✅"
        else:
            result = "Not Duplicate ❌"

    return render_template('index.html', result=result, score=score)


if __name__ == '__main__':
    app.run(debug=True)