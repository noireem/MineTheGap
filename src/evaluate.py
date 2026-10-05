"""Report how well the keyword baseline matches your hand labels.

Usage (from the project root):
    python -m src.evaluate data/seed_passages.csv
    python -m src.evaluate data/passages_labeled.csv

The CSV needs two columns: text, label  (label is Celebrator, Tolerator or Depreciator).
"""
import sys

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.rubric import CLASSES, score_passage


def evaluate(path: str) -> dict:
    df = pd.read_csv(path).dropna(subset=["text", "label"])
    bad = set(df["label"]) - set(CLASSES)
    if bad:
        raise ValueError(f"Unknown labels in {path}: {sorted(bad)}")
    df["pred"] = df["text"].map(lambda t: score_passage(t).label)
    # "Unclear" is never a correct answer, so it counts against the baseline.
    macro_f1 = f1_score(df["label"], df["pred"], labels=CLASSES, average="macro", zero_division=0)
    return {"n": len(df), "macro_f1": macro_f1, "df": df}


def main(path: str) -> None:
    out = evaluate(path)
    df = out["df"]
    print(f"Passages: {out['n']}")
    print(f"Keyword baseline macro-F1: {out['macro_f1']:.2f}\n")
    print(classification_report(df["label"], df["pred"], labels=CLASSES, zero_division=0))
    cm = confusion_matrix(df["label"], df["pred"], labels=CLASSES + ["Unclear"])
    print("Confusion matrix (rows = your label, columns = baseline):")
    print(pd.DataFrame(cm, index=CLASSES + ["Unclear"], columns=CLASSES + ["Unclear"])
          .iloc[:3].to_string())
    if out["n"] < 100:
        print(f"\nNote: only {out['n']} labeled passages. Treat this number as a smoke test, "
              "not a result.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
