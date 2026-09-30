"""Quick checks. Run from the repo root:  python -m pytest"""
from pathlib import Path

import joblib

from src.explain import top_terms
from src.preprocess import clean_text, strip_dateline

MODEL = Path(__file__).resolve().parent.parent / "models" / "fake_news_pipeline.joblib"


def test_dateline_removed():
    assert strip_dateline("WASHINGTON (Reuters) - Senate voted.").strip() == "Senate voted."


def test_clean_text_basic():
    out = clean_text("Visit https://x.com NOW!!! The 3 cats were running (Reuters)")
    assert "http" not in out and "reuters" not in out
    assert "the" not in out.split() and "3" not in out
    assert "cat" in out.split()  # lemmatized


def test_model_predicts_and_explains():
    model = joblib.load(MODEL)
    fake = clean_text("BREAKING watch this video read more image via Getty share now Obama Hillary")
    real = clean_text("Officials said on Tuesday the Senate voted on the bill, a spokesman said on Wednesday.")
    assert model.predict_proba([fake])[0][1] < 0.5
    assert model.predict_proba([real])[0][1] > 0.5
    toward_real, toward_fake = top_terms(model, real, 3)
    assert toward_real  # non-empty


def test_top_terms_empty_text():
    model = joblib.load(MODEL)
    assert top_terms(model, "zzzzqqq") == ([], [])


def test_dateline_only_removes_place_prefix():
    assert strip_dateline("MANCHESTER, England (Reuters) - PM said.") == "PM said."
    sentence = "The cats were running quickly (Reuters) and then stopped."
    assert strip_dateline(sentence) == sentence  # ordinary sentence untouched
