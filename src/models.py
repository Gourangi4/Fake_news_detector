"""Single place for model settings, shared by train.py and the notebook."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TEST_SIZE = 0.2
RANDOM_STATE = 42
DEPLOYED_MODEL = "Logistic Regression"  # used by the Streamlit app (has probabilities + coefficients)


def make_pipeline(clf):
    """TF-IDF + classifier. Vectorizer lives inside the pipeline so it is fit on training data only."""
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def build_models():
    return {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, C=5)),
        "Naive Bayes": make_pipeline(MultinomialNB(alpha=0.1)),
        "Linear SVM": make_pipeline(LinearSVC(C=0.5)),
    }
