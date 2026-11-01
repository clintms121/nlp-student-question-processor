"""Simple NLP processing for the RSU student questions dataset.

Steps:
1. Load the questions from the CSV file.
2. Clean the text (lowercase, remove punctuation).
3. Tokenize (split into words).
4. Remove stopwords (common words like "the", "is", "how").
5. Count word frequencies overall and per category.

Run:  python process_questions.py
Results are printed and saved to the output/ folder.
"""

import csv
import re
from collections import Counter
from pathlib import Path

DATA_FILE = Path("data/rsu_student_questions.csv")
OUTPUT_DIR = Path("output")
EXAMPLE_IDS = ["2", "11", "22"]  # questions shown in the cleaning example

STOPWORDS = {
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "before", "but", "by", "can", "could", "did", "do", "does", "doesnt",
    "dont", "during", "for", "from", "get", "go", "has", "have", "how", "i",
    "if", "im", "in", "into", "is", "isnt", "it", "its", "me", "my", "need",
    "no", "not", "of", "off", "on", "or", "our", "out", "should", "so", "some", "than",
    "that", "the", "their", "there", "these", "they", "this", "to", "up", "was",
    "we", "what", "whats", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_text(text):
    text = text.lower()
    text = text.replace("'", "")             # "what's" -> "whats"
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punctuation
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def process(question):
    cleaned = clean_text(question)
    tokens = tokenize(cleaned)
    keywords = remove_stopwords(tokens)
    return cleaned, tokens, keywords


def main():
    rows = load_questions(DATA_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)

    overall = Counter()
    by_category = {}
    total_tokens = 0
    total_keywords = 0

    with open(OUTPUT_DIR / "processed_questions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "cleaned", "tokens", "keywords"])
        for row in rows:
            cleaned, tokens, keywords = process(row["question"])
            total_tokens += len(tokens)
            total_keywords += len(keywords)
            overall.update(keywords)
            by_category.setdefault(row["category"], Counter()).update(keywords)
            writer.writerow([row["id"], row["category"], row["question"],
                             cleaned, " ".join(tokens), " ".join(keywords)])

    lines = []
    lines.append("=== Dataset summary ===")
    lines.append(f"Questions:              {len(rows)}")
    lines.append(f"Categories:             {len(by_category)}")
    lines.append(f"Tokens before stopwords: {total_tokens}")
    lines.append(f"Tokens after stopwords:  {total_keywords}")
    lines.append(f"Unique keywords:        {len(overall)}")

    lines.append("")
    lines.append("=== Example: text cleaning and tokenization ===")
    for row in (r for r in rows if r["id"] in EXAMPLE_IDS):
        cleaned, tokens, keywords = process(row["question"])
        lines.append(f"Original: {row['question']}")
        lines.append(f"Cleaned:  {cleaned}")
        lines.append(f"Tokens:   {tokens}")
        lines.append(f"Keywords: {keywords}")
        lines.append("")

    lines.append("=== Top 15 keywords (all questions) ===")
    for word, count in overall.most_common(15):
        lines.append(f"{word:<15}{count}")

    lines.append("")
    lines.append("=== Top 5 keywords per category ===")
    for category in sorted(by_category):
        top = ", ".join(f"{w} ({c})" for w, c in by_category[category].most_common(5))
        lines.append(f"{category:<20}{top}")

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "nlp_results.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {OUTPUT_DIR / 'nlp_results.txt'} and {OUTPUT_DIR / 'processed_questions.csv'}")


if __name__ == "__main__":
    main()
