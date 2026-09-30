# Fake News Detector

A machine learning project that classifies a news article as Fake or Real. It uses TF-IDF text features with classic ML models, and includes a Streamlit web app that shows the prediction, a confidence score, and the words that influenced the result.

## Features

- Text cleaning pipeline (URL and dateline removal, stopwords, lemmatization)
- Three models compared: Logistic Regression, Naive Bayes, Linear SVM
- Streamlit app with prediction, confidence and word-level explanation
- Jupyter notebook with data exploration, leakage checks and evaluation
- Saved, ready-to-use model (no training needed to try the app)

## Dataset

ISOT Fake and Real News Dataset (available on Kaggle as "Fake and Real News Dataset").

- `Fake.csv`: 23,481 articles
- `True.csv`: 21,417 articles
- Columns: title, text, subject, date

The data is not included in this repository. To retrain or run the notebook, download it and place `Fake.csv` and `True.csv` in the `data/` folder.

After cleaning, 38,376 articles remain (17,454 fake, 20,922 real). The split is 80/20 and stratified (30,700 train, 7,676 test).

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9914 | 0.9902 | 0.9940 | 0.9921 |
| Naive Bayes | 0.9574 | 0.9633 | 0.9584 | 0.9608 |
| Linear SVM | 0.9922 | 0.9912 | 0.9945 | 0.9928 |

5-fold cross-validation F1 (Logistic Regression): 0.9916 (+/- 0.0005).

The app uses Logistic Regression. It is nearly as accurate as Linear SVM, and it provides probabilities and readable word weights for explanations.

Confusion matrices, a model comparison chart and top-word charts are in `reports/figures/`. Full metrics are in `reports/metrics.json`.

## A note on the high accuracy

About 99% accuracy does not mean fake news detection is solved. Both classes come from a small set of outlets, mostly 2016-2017 US political news. The model learns writing style, for example weekday names and the word "said" for real news, and "via", "video" and "featured image" for fake news. Expect lower accuracy on other outlets, topics or years. This model does not check facts.

Data leakage handled:

- `subject` and `date` columns dropped (`subject` separates the classes almost perfectly).
- The "WASHINGTON (Reuters) -" dateline removed, and the word "reuters" removed everywhere. Nearly all real articles contain it and almost no fake ones do.
- Empty-text rows removed.
- Exact duplicates removed, then near-duplicates (identical after cleaning), so copies of an article cannot appear in both train and test.
- TF-IDF fit on training data only (inside a scikit-learn Pipeline, and inside each cross-validation fold).

## Project structure

```
fake-news-detector/
|-- app.py                       Streamlit app
|-- requirements.txt
|-- LICENSE
|-- data/
|   `-- README.md                Where to put the dataset
|-- models/
|   `-- fake_news_pipeline.joblib    Trained TF-IDF + Logistic Regression
|-- notebooks/
|   `-- fake_news_classification.ipynb   Exploration, training, evaluation
|-- reports/
|   |-- metrics.json
|   `-- figures/                 Confusion matrices, model comparison, top words
|-- src/
|   |-- preprocess.py            Loading and text cleaning
|   |-- models.py                Model and TF-IDF settings
|   |-- train.py                 Train, evaluate, save
|   `-- explain.py               Word-level explanation for a prediction
`-- tests/
    `-- test_basic.py
```

## Getting started

Requires Python 3.11 or newer.

1. Clone the repository and install dependencies

```
git clone <repository-url>
cd fake-news-detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

2. Run the app (a trained model is already included)

```
streamlit run app.py
```

3. Optional: retrain or explore. Download the dataset first (see Dataset above).

```
python -m src.train
jupyter notebook notebooks/fake_news_classification.ipynb
```

4. Optional: run the tests

```
python -m pytest
```

NLTK stopwords and WordNet download automatically the first time the code runs.

## How it works

1. Load `Fake.csv` and `True.csv`, label fake = 0 and real = 1, merge and shuffle.
2. Drop leaky columns, empty rows and duplicates.
3. Combine title and body, then clean: strip dateline, lowercase, remove URLs, punctuation and numbers, remove stopwords (NLTK), lemmatize (WordNet).
4. TF-IDF with unigrams and bigrams, 10,000 features.
5. Train Logistic Regression, Naive Bayes and Linear SVM.
6. Evaluate with accuracy, precision, recall, F1, confusion matrix and 5-fold cross-validation.
7. Explain predictions using logistic regression coefficients multiplied by TF-IDF values.
8. Serve predictions in a Streamlit app.

## Tech stack

Python, pandas, NumPy, scikit-learn, NLTK, matplotlib, seaborn, Streamlit, joblib.

## Limitations and future work

- Learns source style, not facts, and does not verify claims.
- English only, mostly US political news.
- Test on a second dataset (for example WELFake) to measure generalization.
- Try transformer models (for example DistilBERT) for better generalization.

## License

MIT
