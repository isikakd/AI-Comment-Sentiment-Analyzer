from flask import Flask, render_template, request
from services.sentiment_service import analyze_comments

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    results = []
    stats = {"POSITIVE": 0, "NEGATIVE": 0}

    if request.method == "POST":
        text = request.form.get("text", "")
        lang = request.form.get("lang", "tr")
        comments = [c.strip() for c in text.split("\n") if c.strip()]
        results, stats = analyze_comments(comments, lang=lang)

    return render_template("index.html", text=text, results=results, stats=stats)

if __name__ == "__main__":
    app.run(debug=True)