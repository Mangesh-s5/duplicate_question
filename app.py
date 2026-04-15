from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)


model = SentenceTransformer('all-MiniLM-L6-v2')


def get_similarity(q1, q2):
    embeddings = model.encode([q1, q2])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    score = 0

    if request.method == 'POST':
        q1 = request.form['q1']
        q2 = request.form['q2']

        similarity = get_similarity(q1, q2)
        score = round(similarity, 2)

        
        if score >= 0.75:
            result = "Duplicate Question ✅"
        elif score >= 0.5:
            result = "Partially Similar ⚠️"
        else:
            result = "Not Duplicate ❌"

    return render_template('index.html', result=result, score=score)


if __name__ == '__main__':
    app.run(debug=True)