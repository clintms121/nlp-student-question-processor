# nlp-student-question-processor

A simple NLP project that processes 360 student questions from Rogers State University (RSU). Each question has one of 12 categories, such as `admissions`, `financial_aid` and `parking_safety`.

## How to Run

```
pip install -r requirements.txt
python process_questions.py
python train_model.py
```

`process_questions.py` uses only Python's standard library. `train_model.py` needs scikit-learn.

The scripts save three files:

- `output/nlp_results.txt` is the text processing report.
- `output/processed_questions.csv` has every question with its cleaned text, tokens and keywords.
- `output/model_results.txt` is the model report.

## Processing Steps

1. **Load** the questions from `data/rsu_student_questions.csv`.
2. **Clean** the text by making it lowercase and removing punctuation.
3. **Tokenize** the text by splitting it into words.
4. **Remove stopwords**, which are common words like "how", "do", "I" and "the".
5. **Count** how often each keyword appears, overall and in each category.

---

## Section 3: Evidence of NLP Processing

### Evidence 1: Text Cleaning and Tokenization

| Original Question | Tokens | Keywords (stopwords removed) |
|---|---|---|
| I lost my room key. What do I do? | i, lost, my, room, key, what, do, i, do | lost, room, key |
| my mic doesnt work in my online class meeting, how do i fix it | my, mic, doesnt, work, in, my, online, class, meeting, how, do, i, fix, it | mic, work, online, class, meeting, fix |
| What's the difference between a BS and a BAS degree? | whats, the, difference, between, a, bs, and, a, bas, degree | difference, between, bs, bas, degree |

**Explanation:** This table shows real questions from my dataset after cleaning, tokenizing and removing stopwords. The program removed punctuation and capital letters and kept only the important words.

### Evidence 2: Dataset Summary and Top Keywords

```
=== Dataset summary ===
Questions:              360
Categories:             12
Tokens before stopwords: 3273
Tokens after stopwords:  1527
Unique keywords:        699

=== Top 15 keywords (all questions) ===
campus         48
rsu            36
student        24
online         23
students       23
class          23
library        15
apply          14
find           13
semester       13
course         13
help           13
classes        12
aid            12
parking        11
```

**Explanation:** Removing stopwords cut the word count from 3,273 to 1,527, so more than half of the words were filler. The most common keywords are "campus", "rsu", "student" and "online", which shows that most questions are about campus services and online classes.

### Evidence 3: Top Keywords per Category

| Category | Top 5 Keywords (count) |
|---|---|
| academic_advising | academic (5), degree (4), course (3), major (3), take (3) |
| admissions | apply (8), rsu (6), school (4), application (4), state (3) |
| billing_payments | pay (8), bill (8), tuition (7), online (4), payment (4) |
| campus_life | student (7), students (6), campus (6), rsu (6), club (4) |
| career_services | career (7), job (5), rsu (4), find (4), help (3) |
| financial_aid | aid (10), rsu (8), financial (7), scholarships (4), fafsa (4) |
| health_wellness | students (6), campus (5), counseling (4), support (3), help (3) |
| housing_dining | campus (10), housing (8), room (6), meal (4), live (3) |
| library_tutoring | library (15), tutoring (5), book (4), online (3), study (3) |
| parking_safety | campus (13), parking (10), permit (4), report (3), car (3) |
| registration | class (14), enroll (8), course (6), credit (4), semester (4) |
| technology_support | online (7), rsu (6), email (6), student (4), campus (4) |

**Explanation:** Each category has its own keywords, such as "parking" and "permit" for parking_safety and "aid" and "fafsa" for financial_aid. This means the keywords can help a program guess the category of a new question.

---

## Section 4: Current Model or Method Result

### Method

I used a Naive Bayes classifier to predict the category of each question. First, the program cleaned each question and removed stopwords. Then it turned the keywords into numbers with TF-IDF. I trained the model on 288 questions (80%) and tested it on 72 questions (20%).

### Actual Output

```
=== Model: TF-IDF + Multinomial Naive Bayes ===
Training questions: 288
Testing questions:  72
Accuracy:           63.9%

                    precision    recall  f1-score   support
 academic_advising       0.67      0.67      0.67         6
        admissions       0.00      0.00      0.00         6
  billing_payments       0.62      0.83      0.71         6
       campus_life       0.50      0.50      0.50         6
   career_services       0.75      0.50      0.60         6
     financial_aid       0.50      0.67      0.57         6
   health_wellness       0.33      0.17      0.22         6
    housing_dining       0.75      1.00      0.86         6
  library_tutoring       1.00      0.83      0.91         6
    parking_safety       0.71      0.83      0.77         6
      registration       0.75      1.00      0.86         6
technology_support       0.57      0.67      0.62         6
```

The model predicted the correct category for 63.9% of the test questions. Random guessing with 12 categories would only be about 8% correct.

### What Is Working

The model works well when a category has clear keywords. It got all 6 test questions right for registration and for housing_dining. For example, it correctly labeled "How do I cancel my housing contract?" as housing_dining.

### What Needs Improvement

The model got all 6 admissions test questions wrong. These questions use words that also appear in other categories, like "tuition", "semester" and "credit". For example, it labeled "How is my residency decided for in-state tuition?" as billing_payments. The model also has only 24 training questions per category, which is not much data.

---

## Section 5: Next Steps

1. Add more questions to the dataset, especially for admissions and health_wellness.
2. Combine word forms like "student" and "students" using lemmatization.
3. Add two-word phrases like "financial aid" and "parking permit" as features.
4. Try other models, like Logistic Regression and Linear SVM, and compare their accuracy.
5. Use cross-validation so the results do not depend on one random split of 72 test questions.
