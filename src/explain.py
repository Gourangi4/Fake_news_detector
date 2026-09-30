"""Explain a prediction using logistic regression coefficients."""
import numpy as np


def top_terms(pipeline, cleaned_text: str, n: int = 8):
    """Return (toward_real, toward_fake) lists of (word, weight) present in this text.

    weight = tfidf value * coefficient. Positive pushes toward REAL (label 1),
    negative pushes toward FAKE (label 0).
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    vec = tfidf.transform([cleaned_text])
    names = np.array(tfidf.get_feature_names_out())
    idx = vec.nonzero()[1]
    if len(idx) == 0:
        return [], []
    contrib = vec[0, idx].toarray().ravel() * clf.coef_[0][idx]
    order = np.argsort(contrib)
    fake = [(names[idx[i]], float(-contrib[i])) for i in order[:n] if contrib[i] < 0]
    real = [(names[idx[i]], float(contrib[i])) for i in order[::-1][:n] if contrib[i] > 0]
    return real, fake
