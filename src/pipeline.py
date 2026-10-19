"""Command line for the pipeline. Run from the project root:

    python -m src.pipeline build                       # 10-Ks -> inventory + candidate DEI passages
    python -m src.pipeline revenue --email nemeyers@aggies.ncat.edu   # SEC XBRL -> annual revenue
    python -m src.pipeline evaluate                    # compare models, grouped CV by company
    python -m src.pipeline link                        # company scores + Spearman vs volatility
"""
import argparse
import json
import sys

import pandas as pd

from src import config
from src.data_loader import UNIVERSE, build_candidates, build_inventory, validate_universe
from src.evaluate import cross_validate, inter_annotator_kappa, primary_labels, save_figures
from src.market import build_revenue_table, link_score_to_volatility, volatility_table
from src.nlp_engine import MODEL_NAMES, company_year_scores, score_candidates_grouped


def _require_csv(path, what: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"Missing {path}\n"
            f"{what}\n"
            f"Label rows from {config.CANDIDATES} (columns: company,fiscal_year,section,text,label,labeler)."
        )
    return pd.read_csv(path)


def run_linkage(model_name: str, first_year: int, last_year: int) -> dict:
    cands = _require_csv(config.CANDIDATES, "Run `python -m src.pipeline build` first.")
    labeled = primary_labels(_require_csv(
        config.LABELED,
        "Evaluate/link need hand labels. Copy ~100–150 candidate passages into this file.",
    ))
    revenue = pd.read_csv(config.REVENUE_CSV)
    out, merged_all = {}, []
    for name in ["keyword", model_name]:
        scores = company_year_scores(cands, score_candidates_grouped(cands, labeled, name))
        scores.insert(0, "model", name)
        company_level = scores.groupby("company", as_index=False)["dei_score"].mean()
        merged = company_level.merge(volatility_table(revenue, first_year, last_year), on="company")
        merged.insert(0, "model", name)
        try:
            out[name] = link_score_to_volatility(merged)
        except ValueError as e:          # e.g. constant scores: record why, keep the other model's result
            out[name] = {"error": str(e)}
        merged_all.append((scores, merged))
    config.SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([s for s, _ in merged_all]).to_csv(config.SCORES_CSV, index=False)
    pd.concat([m for _, m in merged_all]).to_csv(config.MERGED_CSV, index=False)
    return out


def cli(argv: list) -> None:
    p = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="inventory the downloaded 10-Ks and extract candidate DEI passages")
    r = sub.add_parser("revenue", help="download annual revenue from SEC XBRL")
    r.add_argument("--email", required=True, help="sent in the User-Agent header, as SEC requires")
    e = sub.add_parser("evaluate", help="grouped cross-validation of all models")
    e.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    lk = sub.add_parser("link", help="company scores and Spearman vs revenue volatility")
    lk.add_argument("--model", default="embedding_lr", choices=MODEL_NAMES[1:])
    lk.add_argument("--first-year", type=int, default=2019)
    lk.add_argument("--last-year", type=int, default=2025)
    a = p.parse_args(argv)

    problems = validate_universe(UNIVERSE)
    if problems:
        sys.exit("Invalid company universe:\n- " + "\n- ".join(problems))

    if a.cmd == "build":
        inv = build_inventory(config.FILINGS_DIR)
        config.INVENTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(config.INVENTORY_CSV, index=False)
        print(inv.drop(columns='path').to_string(index=False), '\n')
        df = build_candidates()
        config.CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.CANDIDATES, index=False)
        table = df.groupby(["company", "fiscal_year"]).size().unstack(fill_value=0)
        print(table.to_string())
        for t in [u["ticker"] for u in UNIVERSE]:
            for fy in (2020, 2024):
                if t not in table.index or fy not in table.columns or table.loc[t, fy] == 0:
                    print(f"WARNING: no passages for {t} FY{fy} (missing filing, wrong year, or no DEI text)")
        print(f"\n{len(df)} candidate passages -> {config.CANDIDATES}\n"
              f"Copy label-worthy rows into {config.LABELED} (columns: company,fiscal_year,section,text,label,labeler).")
    elif a.cmd == "revenue":
        df = build_revenue_table(f"MineTheGap student project {a.email}")
        config.REVENUE_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.REVENUE_CSV, index=False)
        pivot = df.pivot(index="company", columns="fiscal_year", values="revenue_usd")
        print(pivot.to_string())
        for t in [u["ticker"] for u in UNIVERSE]:
            for fy in range(2019, 2026):
                missing = t not in pivot.index or fy not in pivot.columns or pd.isna(pivot.loc[t, fy])
                if missing:
                    print(f"WARNING: missing revenue for {t} FY{fy}")
    elif a.cmd == "evaluate":
        labeled = _require_csv(
            config.LABELED,
            "Evaluate/link need hand labels. Copy ~100–150 candidate passages into this file.",
        )
        df = primary_labels(labeled)
        results = cross_validate(df, a.models)
        results["_inter_annotator"] = inter_annotator_kappa(labeled)
        config.RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        config.RESULTS_JSON.write_text(json.dumps(results, indent=2))
        save_figures(results)
        print(f"{len(df)} labeled passages, {df['company'].nunique()} companies"
              + ("  <- fewer than 100: treat as a smoke test" if len(df) < 100 else ""))
        for name in a.models:
            r = results[name]
            if "skipped" in r:
                print(f"{name:18s} SKIPPED  {r['skipped']}")
            else:
                print(f"{name:18s} macro-F1 {r['macro_f1']:.2f}  95% CI "
                      f"[{r['macro_f1_ci95'][0]:.2f}, {r['macro_f1_ci95'][1]:.2f}]  "
                      f"kappa {r['cohen_kappa_vs_human']:.2f}  meets target: {r.get('meets_target')}")
        print("Inter-annotator:", results["_inter_annotator"])
    elif a.cmd == "link":
        out = run_linkage(a.model, a.first_year, a.last_year)
        existing = json.loads(config.RESULTS_JSON.read_text()) if config.RESULTS_JSON.exists() else {}
        existing["linkage"] = out
        config.RESULTS_JSON.write_text(json.dumps(existing, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cli(sys.argv[1:])
