"""Evaluation: grouped cross-validation of all models, macro-F1, bootstrap CI, Cohen's kappa.

Also keeps the original baseline report:

Usage (from the project root):
    python -m src.evaluate data/seed_passages.csv
    python -m src.evaluate data/passages_labeled.csv

The CSV needs two columns: text, label  (label is Celebrator, Tolerator or Depreciator).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src import config
from src.nlp_engine import make_model
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


def primary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid labels from the main labeler only, so double-labeled rows are not counted twice."""
    df = df[df["label"].isin(CLASSES)].copy()
    if "labeler" in df and df["labeler"].notna().any() and df["labeler"].nunique() > 1:
        df = df[df["labeler"] == df["labeler"].value_counts().idxmax()]
    return df.drop_duplicates(subset=["company", "fiscal_year", "text"]).reset_index(drop=True)


def group_folds(df: pd.DataFrame, n_splits: int = 5) -> list:
    """(train_idx, test_idx) pairs where each company appears in exactly one test fold."""
    from sklearn.model_selection import GroupKFold
    n = min(n_splits, df["company"].nunique())
    if n < 2:
        raise ValueError("Need passages from at least 2 companies for grouped cross-validation.")
    return list(GroupKFold(n_splits=n).split(df, df["label"], df["company"]))


def bootstrap_f1_ci(y_true, y_pred, n_boot: int = 500, seed: int = 0) -> tuple:
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[i], y_pred[i], labels=CLASSES, average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cross_validate(df: pd.DataFrame, model_names: list, n_splits: int = 5) -> dict:
    """Same folds, same metric for every model. Models that cannot load are reported as skipped."""
    from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
    folds = group_folds(df, n_splits)
    results = {}
    for name in model_names:
        pred = np.empty(len(df), dtype=object)
        try:
            for tr, te in folds:
                model = make_model(name).fit(df["text"].iloc[tr], df["label"].iloc[tr])
                pred[te] = model.predict(df["text"].iloc[te])
        except (ImportError, OSError) as e:          # missing package or no network for model download
            results[name] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        labels = CLASSES + ["Unclear"]
        lo, hi = bootstrap_f1_ci(df["label"], pred)
        results[name] = {
            "n": int(len(df)), "folds": len(folds),
            # "Unclear" is never correct, so the keyword baseline is penalized for abstaining.
            "macro_f1": float(f1_score(df["label"], pred, labels=CLASSES, average="macro", zero_division=0)),
            "macro_f1_ci95": [lo, hi],
            "cohen_kappa_vs_human": float(cohen_kappa_score(df["label"], pred)),
            "report": classification_report(df["label"], pred, labels=CLASSES, zero_division=0,
                                            output_dict=True),
            "confusion": confusion_matrix(df["label"], pred, labels=labels).tolist(),
            "confusion_labels": labels,
        }
    base = results.get("keyword", {}).get("macro_f1")
    for r in results.values():
        if "macro_f1" in r and base is not None:
            # Proposal's success rule: macro-F1 >= 0.60 and >= 0.10 above the keyword baseline.
            r["meets_target"] = bool(r["macro_f1"] >= 0.60 and r["macro_f1"] - base >= 0.10)
    return results


def inter_annotator_kappa(df: pd.DataFrame):
    """Cohen's kappa between two labelers on passages both labeled. None if not enough overlap."""
    from sklearn.metrics import cohen_kappa_score
    df = df[df["label"].isin(CLASSES)]
    if "labeler" not in df or df["labeler"].nunique() < 2:
        return None
    wide = df.pivot_table(index=["company", "fiscal_year", "text"], columns="labeler",
                          values="label", aggfunc="first").dropna()
    if len(wide) < 2 or wide.shape[1] < 2:
        return None
    a, b = wide.columns[:2]
    return {"n": int(len(wide)), "labelers": [str(a), str(b)],
            "kappa": float(cohen_kappa_score(wide[a], wide[b]))}


def save_figures(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    ran = {k: v for k, v in results.items() if isinstance(v, dict) and "macro_f1" in v}
    if ran:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = list(ran)
        f1 = [ran[n]["macro_f1"] for n in names]
        err = [[ran[n]["macro_f1"] - ran[n]["macro_f1_ci95"][0] for n in names],
               [ran[n]["macro_f1_ci95"][1] - ran[n]["macro_f1"] for n in names]]
        ax.bar(names, f1, yerr=err, capsize=4)
        ax.set_ylabel("Macro-F1 (grouped CV)")
        ax.set_ylim(0, 1)
        ax.set_title("Model comparison, same folds")
        fig.tight_layout()
        fig.savefig(config.FIG_DIR / "model_comparison.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])