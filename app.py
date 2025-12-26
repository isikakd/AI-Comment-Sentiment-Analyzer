from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text = ""   

    if request.method == "POST":
        text = request.form["text"]
        vector = vectorizer.transform([text])
        sentiment = model.predict(vector)[0]

    return render_template(
        "index.html",
        sentiment=sentiment,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)