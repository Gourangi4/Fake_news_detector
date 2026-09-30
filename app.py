"""Streamlit app: paste a news article, get Fake/Real with confidence and key words."""
from pathlib import Path

import joblib
import streamlit as st

from src.explain import top_terms
from src.preprocess import clean_text

MODEL_PATH = Path(__file__).parent / "models" / "fake_news_pipeline.joblib"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    load_model()
except Exception as exc:  # missing file or scikit-learn version mismatch
    st.error(
        "Could not load the trained model. Make sure models/fake_news_pipeline.joblib exists "
        "and that scikit-learn matches requirements.txt. To rebuild it, put Fake.csv and "
        "True.csv in data/ and run: python -m src.train"
    )
    st.exception(exc)
    st.stop()


SAMPLES = {
    "Sample: wire-style report": (
        "WASHINGTON (Reuters) - The U.S. Senate on Tuesday voted to approve a spending bill "
        "that would fund the federal government through March, sending the measure to the "
        "House of Representatives, officials said on Wednesday."
    ),
    "Sample: clickbait-style post": (
        "BREAKING: You won't believe what Obama just did! Watch this video before it gets "
        "deleted. Read more and share this image via our page. Featured image via Getty."
    ),
}

st.title("Fake News Detector")
st.caption("TF-IDF + Logistic Regression trained on the ISOT Fake and Real News dataset.")

choice = st.selectbox("Try a sample or paste your own", ["Paste my own"] + list(SAMPLES))
default = SAMPLES.get(choice, "")
text = st.text_area("Article text (headline + body works best)", value=default, height=220)

if st.button("Predict", type="primary"):
    if len(text.split()) < 15:
        st.warning("Please enter at least a few sentences (15+ words) for a reliable result.")
    else:
        model = load_model()
        cleaned = clean_text(text)
        proba_real = float(model.predict_proba([cleaned])[0][1])
        is_real = proba_real >= 0.5
        confidence = min(proba_real if is_real else 1 - proba_real, 0.999)  # never show 100%

        if is_real:
            st.success(f"Prediction: REAL  ({confidence:.1%} confidence)")
        else:
            st.error(f"Prediction: FAKE  ({confidence:.1%} confidence)")
        st.progress(proba_real, text=f"Probability real: {proba_real:.1%}")

        real_terms, fake_terms = top_terms(model, cleaned)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Words pushing toward REAL**")
            for w, s in real_terms or [("none found", 0)]:
                st.write(f"- {w}")
        with c2:
            st.markdown("**Words pushing toward FAKE**")
            for w, s in fake_terms or [("none found", 0)]:
                st.write(f"- {w}")

st.divider()
st.caption(
    "Limits: the model learned writing style of the sources in this dataset (mostly 2016-2017 "
    "US political news). It does not check facts, and it will be less reliable on other "
    "topics, years or outlets. Use it as a demo, not as a truth detector."
)
