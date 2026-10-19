"""Classifiers for DEI language (Celebrator / Tolerator / Depreciator) and the company-year score.

Every model exposes fit / predict / predict_proba, with probability columns in CLASSES order,
so evaluate.py can compare them under identical conditions.
"""
import numpy as np
import pandas as pd

from src.rubric import CLASSES, score_passage

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


FINBERT_MODEL = "ProsusAI/finbert"


MODEL_NAMES = ["keyword", "tfidf_lr", "embedding_lr", "finbert_zero_shot"]


def _to_class_order(proba: np.ndarray, classes) -> np.ndarray:
    """Reorder (and zero-fill) probability columns to [Celebrator, Tolerator, Depreciator]."""
    out = np.zeros((proba.shape[0], len(CLASSES)))
    for j, c in enumerate(classes):
        if c in CLASSES:
            out[:, CLASSES.index(c)] = proba[:, j]
    return out


class KeywordBaseline:
    """The non-AI baseline: src/rubric.py cue counts. No training."""
    name = "keyword"

    def fit(self, texts, y):
        return self

    def predict(self, texts):
        return np.array([score_passage(t).label for t in texts], dtype=object)

    def predict_proba(self, texts):
        rows = []
        for t in texts:
            c = score_passage(t).counts
            total = sum(c.values())
            rows.append([c[k] / total if total else 0.0 for k in CLASSES])
        return np.array(rows)


class TfidfLR:
    """Trained from scratch on the labeled passages: TF-IDF (1-2 grams) + logistic regression."""
    name = "tfidf_lr"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def fit(self, texts, y):
        self.pipe.fit(list(texts), list(y))     # vectorizer is fit on the training fold only
        return self

    def predict(self, texts):
        return self.pipe.predict(list(texts))

    def predict_proba(self, texts):
        return _to_class_order(self.pipe.predict_proba(list(texts)), self.pipe.classes_)


_ST_MODEL = None


def _embed(texts) -> np.ndarray:
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(EMBED_MODEL)
    return _ST_MODEL.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


class EmbeddingLR:
    """Pretrained sentence embeddings (frozen, not fine-tuned) + logistic regression head."""
    name = "embedding_lr"

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, texts, y):
        self.lr.fit(_embed(texts), list(y))
        return self

    def predict(self, texts):
        return self.lr.predict(_embed(texts))

    def predict_proba(self, texts):
        return _to_class_order(self.lr.predict_proba(_embed(texts)), self.lr.classes_)


class FinBERTZeroShot:
    """Off-the-shelf FinBERT tone, no training. positive->Celebrator, neutral->Tolerator,
    negative->Depreciator. A deliberately crude mapping: it tests whether generic financial
    tone already separates the classes."""
    name = "finbert_zero_shot"
    _MAP = {"positive": "Celebrator", "neutral": "Tolerator", "negative": "Depreciator"}
    _pipe = None

    def fit(self, texts, y):
        return self

    def predict_proba(self, texts):
        if FinBERTZeroShot._pipe is None:
            from transformers import pipeline
            FinBERTZeroShot._pipe = pipeline("text-classification", model=FINBERT_MODEL,
                                             truncation=True, max_length=512, top_k=None)
        rows = []
        for scores in FinBERTZeroShot._pipe(list(texts)):
            by_class = {self._MAP[s["label"].lower()]: s["score"] for s in scores}
            rows.append([by_class.get(c, 0.0) for c in CLASSES])
        return np.array(rows)

    def predict(self, texts):
        p = self.predict_proba(texts)
        return np.array([CLASSES[i] for i in p.argmax(axis=1)], dtype=object)


def make_model(name: str):
    return {"keyword": KeywordBaseline, "tfidf_lr": TfidfLR,
            "embedding_lr": EmbeddingLR, "finbert_zero_shot": FinBERTZeroShot}[name]()


def score_candidates_grouped(cands: pd.DataFrame, labeled: pd.DataFrame, model_name: str) -> np.ndarray:
    """Class probabilities for every candidate passage. Trained models predict each company's
    passages using a model fit only on OTHER companies' labeled passages (no in-sample scoring)."""
    from sklearn.model_selection import GroupKFold
    proba = np.zeros((len(cands), len(CLASSES)))
    n = min(5, cands["company"].nunique())
    for _, te in GroupKFold(n_splits=n).split(cands, groups=cands["company"]):
        test_companies = set(cands["company"].iloc[te])
        train = labeled[~labeled["company"].isin(test_companies)]
        model = make_model(model_name).fit(train["text"], train["label"])
        proba[te] = model.predict_proba(cands["text"].iloc[te])
    return proba


def company_year_scores(cands: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """DEI Sentiment Score = mean[P(Celebrator) - P(Depreciator)] per company-year, in [-1, 1]."""
    df = cands[["company", "fiscal_year"]].copy()
    for j, c in enumerate(CLASSES):
        df[f"p_{c}"] = proba[:, j]
    df["score"] = df["p_Celebrator"] - df["p_Depreciator"]
    g = df.groupby(["company", "fiscal_year"])
    out = g[["score"] + [f"p_{c}" for c in CLASSES]].mean()
    out["n_passages"] = g.size()
    p = out[[f"p_{c}" for c in CLASSES]].to_numpy()
    out["class"] = [CLASSES[i] if p[k].sum() > 0 else "Unclear" for k, i in enumerate(p.argmax(axis=1))]
    return out.reset_index().rename(columns={"score": "dei_score"})


def shift_table(scores: pd.DataFrame, start: int = 2020, end: int = 2024) -> pd.DataFrame:
    """FY start -> FY end change in score and in number of DEI passages."""
    a = scores[scores.fiscal_year == start].set_index("company")
    b = scores[scores.fiscal_year == end].set_index("company")
    idx = a.index.intersection(b.index)
    return pd.DataFrame({"score_change": b.loc[idx, "dei_score"] - a.loc[idx, "dei_score"],
                         "passage_change": b.loc[idx, "n_passages"] - a.loc[idx, "n_passages"]}).reset_index()