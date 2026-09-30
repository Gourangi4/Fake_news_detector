"""Text cleaning and data loading for the fake news classifier."""
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
STOP = set(stopwords.words("english"))
LEMMA = WordNetLemmatizer()
_lemma_cache = {}

# "WASHINGTON (Reuters) - " style prefix at the start of real articles. It must start with a
# capitalised place name, so ordinary sentences that mention (Reuters) are left alone.
# The model would cheat on this, so it is removed.
DATELINE = re.compile(r"^\s*[A-Z]{2,}[A-Za-z0-9 ,.\-/'\u2019]{0,60}?\(Reuters\)\s*[-\u2013\u2014]*\s*")
URL = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHA = re.compile(r"[^a-z\s]")
# Source names that give away the label without saying anything about content.
SOURCE_WORDS = {"reuters"}


def strip_dateline(text: str) -> str:
    return DATELINE.sub("", text, count=1)


def clean_text(text: str) -> str:
    """Lowercase, drop dateline/URLs/punctuation/numbers/stopwords, lemmatize."""
    text = strip_dateline(str(text))
    text = URL.sub(" ", text.lower())
    text = NON_ALPHA.sub(" ", text)
    out = []
    for tok in text.split():
        if len(tok) < 2 or tok in STOP or tok in SOURCE_WORDS:
            continue
        lem = _lemma_cache.get(tok)
        if lem is None:
            lem = LEMMA.lemmatize(tok)
            _lemma_cache[tok] = lem
        out.append(lem)
    return " ".join(out)


def load_raw(data_dir=DATA_DIR) -> pd.DataFrame:
    """Load Fake.csv and True.csv, label them (fake=0, real=1), merge, shuffle."""
    fake_path, real_path = Path(data_dir) / "Fake.csv", Path(data_dir) / "True.csv"
    for path in (fake_path, real_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the ISOT 'Fake and Real News Dataset' from Kaggle "
                "and put Fake.csv and True.csv in the data/ folder (see data/README.md)."
            )
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake["label"] = 0
    real["label"] = 1
    df = pd.concat([fake, real], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leaky columns, empty rows and duplicates, then build the clean text column."""
    df = df.drop(columns=["subject", "date"], errors="ignore").copy()
    df["raw"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    df = df[df["text"].fillna("").str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["title", "text"])
    df["clean"] = df["raw"].map(clean_text)
    df = df[df["clean"].str.len() > 0]
    # Near-duplicates: different raw text that becomes identical after cleaning.
    # Keeping them would put copies of the same article in both train and test.
    df = df.drop_duplicates(subset=["clean"])
    return df.reset_index(drop=True)
