"""Simple category classifier for the RSU student questions dataset.

Method: TF-IDF word features + Multinomial Naive Bayes.
The questions are cleaned with the same steps from process_questions.py.

Run:  python train_model.py
Results are printed and saved to output/model_results.txt.
"""

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from process_questions import DATA_FILE, OUTPUT_DIR, load_questions, process


def main():
    rows = load_questions(DATA_FILE)
    texts = [" ".join(process(r["question"])[2]) for r in rows]  # keywords only
    labels = [r["category"] for r in rows]
    questions = [r["question"] for r in rows]

    # 80% train, 20% test, same number of test questions from each category
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(
        texts, labels, questions, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer()
    model = MultinomialNB()
    model.fit(vectorizer.fit_transform(X_train), y_train)
    predictions = model.predict(vectorizer.transform(X_test))

    lines = []
    lines.append("=== Model: TF-IDF + Multinomial Naive Bayes ===")
    lines.append(f"Training questions: {len(X_train)}")
    lines.append(f"Testing questions:  {len(X_test)}")
    lines.append(f"Accuracy:           {accuracy_score(y_test, predictions):.1%}")

    lines.append("")
    lines.append("=== Results per category ===")
    lines.append(classification_report(y_test, predictions, zero_division=0))

    lines.append("=== Example predictions ===")
    for q, actual, pred in list(zip(q_test, y_test, predictions))[:8]:
        mark = "OK   " if actual == pred else "WRONG"
        lines.append(f"[{mark}] {q}")
        lines.append(f"        actual: {actual} | predicted: {pred}")

    lines.append("")
    lines.append("=== Most common mistakes (actual -> predicted) ===")
    mistakes = Counter((a, p) for a, p in zip(y_test, predictions) if a != p)
    for (actual, pred), count in mistakes.most_common(5):
        lines.append(f"{actual} -> {pred}: {count}")

    report = "\n".join(lines)
    print(report)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "model_results.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {OUTPUT_DIR / 'model_results.txt'}")


if __name__ == "__main__":
    main()
