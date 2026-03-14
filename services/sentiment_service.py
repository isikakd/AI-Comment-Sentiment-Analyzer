from transformers import pipeline

sentiment_tr = pipeline(
    "sentiment-analysis",
    model="savasy/bert-base-turkish-sentiment-cased"
)

sentiment_en = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def detect_language(text):
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    turkish_words = {
        "ve", "bir", "bu", "da", "de", "ile", "için", "çok",
        "ama", "fakat", "güzel", "iyi", "kötü", "var", "yok",
        "mi", "mı", "mu", "mü", "ne", "en", "daha", "hiç",
        "gayet", "harika", "berbat", "memnun", "ürün"
    }
    if any(ch in turkish_chars for ch in text):
        return "tr"
    words = set(text.lower().split())
    if words & turkish_words:
        return "tr"
    return "en"

def normalize_label(label):
    if "POSITIVE" in label.upper() or label.upper() == "1":
        return "POSITIVE"
    return "NEGATIVE"

def analyze_comment(comment, lang="auto"):
    detected = detect_language(comment) if lang == "auto" else lang
    model = sentiment_tr if detected == "tr" else sentiment_en
    prediction = model(comment, truncation=True, max_length=512)[0]
    return {
        "text": comment,
        "sentiment": normalize_label(prediction["label"]),
        "confidence": round(prediction["score"], 3),
        "lang": detected
    }

def analyze_comments(comments, lang="auto"):
    results = []
    stats = {"POSITIVE": 0, "NEGATIVE": 0}
    for comment in comments:
        if not comment.strip():
            continue
        result = analyze_comment(comment, lang=lang)
        stats[result["sentiment"]] += 1
        results.append(result)
    return results, stats