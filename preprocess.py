import re
import json
import os
from typing import List, Iterable
from langdetect import detect, DetectorFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Ensure deterministic language detection
DetectorFactory.seed = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load resources
with open(os.path.join(DATA_DIR, 'stopwords_id.txt'), 'r', encoding='utf-8') as f:
    STOPWORDS_ID = set([w.strip() for w in f if w.strip()])

# Slang and abbreviation maps
with open(os.path.join(DATA_DIR, 'slang_indonesia.json'), 'r', encoding='utf-8') as f:
    SLANG_MAP = json.load(f)
with open(os.path.join(DATA_DIR, 'abbreviations.json'), 'r', encoding='utf-8') as f:
    ABBR_MAP = json.load(f)

# Build stemmer
_factory = StemmerFactory()
_STEMMER = _factory.create_stemmer()

# Regex patterns
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@[\w_]+")
HASHTAG_RE = re.compile(r"#[\w_]+")
EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",
    flags=re.UNICODE,
)
NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_token(tok: str) -> str:
    """Normalize slang/abbrev, keep Indonesian focus."""
    if not tok:
        return tok
    # Priority: slang, then abbreviation
    if tok in SLANG_MAP:
        tok = SLANG_MAP[tok]
    elif tok in ABBR_MAP:
        tok = ABBR_MAP[tok]
    return tok


def basic_clean(text: str) -> str:
    if not text:
        return ''
    text = text.lower()
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = HASHTAG_RE.sub(' ', text)
    text = EMOJI_RE.sub(' ', text)
    # keep spaces and alnum only
    text = NONALNUM_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text).strip()
    return text


def normalize_text(text: str) -> str:
    """Full normalization pipeline for Indonesian text (slang/abbr aware)."""
    cleaned = basic_clean(text)
    # language detection (best-effort); proceed regardless
    try:
        _ = detect(cleaned)  # value unused; we still process same way
    except Exception:
        pass
    tokens = cleaned.split()
    # map slang/abbr
    norm_tokens = [normalize_token(t) for t in tokens]
    # remove stopwords and single letters (mostly noise)
    filtered = [t for t in norm_tokens if t and t not in STOPWORDS_ID and len(t) > 1]
    # stem Indonesian words
    stemmed = [_STEMMER.stem(t) for t in filtered]
    return ' '.join(stemmed)


def tokenize_for_ml(doc: str) -> List[str]:
    """Tokenizer for scikit-learn vectorizer. Returns tokens list."""
    norm = normalize_text(doc)
    return norm.split()


def flatten(iterable_of_texts: Iterable[str]) -> str:
    return '\n'.join([t for t in iterable_of_texts if t])


# WordCloud helper
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

def generate_wordcloud(texts: Iterable[str], stopwords: set, out_path: str, colormap: str = 'viridis') -> None:
    text_blob = flatten(texts)
    if not text_blob.strip():
        # create an empty placeholder
        wc = WordCloud(width=800, height=400, background_color='white')
        img = wc.generate('tidak ada data').to_image()
        img.save(out_path)
        return
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        collocations=False,
        colormap=colormap,
    )
    wc.generate(text_blob)
    # Save directly
    wc.to_file(out_path)
