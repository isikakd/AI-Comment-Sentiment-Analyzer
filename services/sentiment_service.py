from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")


def analyze_comment(comment):
    prediction = sentiment_model(comment)[0]

    return {
        "text": comment,
        "sentiment": prediction["label"],
        "confidence": round(prediction["score"], 3)
    }


def analyze_comments(comments):
    results = []
    stats = {
        "POSITIVE": 0,
        "NEGATIVE": 0
    }

    for comment in comments:
        if not comment.strip():
            continue

        result = analyze_comment(comment)

        stats[result["sentiment"]] += 1
        results.append(result)

    return results, stats