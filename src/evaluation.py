"""
Evaluation suite. Used by eval/run_evaluation.py (writes the results to files)
and by the Evaluation tab of the app (shows the same numbers live).

Ground truth files in eval/ were labeled by hand by the team:
  test_queries.csv          25 free-text queries -> relevant documents
  recommendation_cases.csv  10 seed documents   -> documents a user would expect
  context_tests.csv          6 ambiguous queries with a previous search in the session
  chatbot_tests.csv         17 questions (12 in scope with expected source, 5 out of scope)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .recommender import Recommender, SearchContext

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def _ids(s):
    s = "" if pd.isna(s) else str(s)
    return {int(x) for x in s.split(";") if x.strip()}


def load_queries(path=EVAL_DIR / "test_queries.csv"):
    q = pd.read_csv(path)
    q["relevant"] = q["relevant_ids"].apply(_ids)
    return q


def retrieval_metrics(ranked_ids, relevant, k=5):
    top = ranked_ids[:k]
    hits = [d for d in top if d in relevant]
    rr = next((1 / (i + 1) for i, d in enumerate(ranked_ids) if d in relevant), 0.0)
    return {
        f"P@{k}": len(hits) / k,
        f"R@{k}": len(hits) / len(relevant),
        "Hit@1": 1.0 if ranked_ids and ranked_ids[0] in relevant else 0.0,
        "MRR": rr,
    }


def evaluate_search(search_fn, queries, k=5, text_col="query"):
    """search_fn(text) -> ranked doc ids. Returns per-query table and averages."""
    rows = []
    for _, q in queries.iterrows():
        ranked = search_fn(q[text_col])
        m = retrieval_metrics(ranked, q["relevant"], k)
        rows.append({"id": q.iloc[0], "input": q[text_col],
                     "relevant": ", ".join(str(x) for x in sorted(q["relevant"])),
                     "top5": ", ".join(str(x) for x in ranked[:k]), **m})
    per_query = pd.DataFrame(rows)
    summary = per_query[[f"P@{k}", f"R@{k}", "Hit@1", "MRR"]].mean().round(3).to_dict()
    return per_query, summary


def max_precision_at_k(relevant_sets, k=5):
    # with 1-4 relevant docs per query, P@5 cannot reach 1.0; this is the ceiling
    return round(float(np.mean([min(len(r), k) / k for r in relevant_sets])), 3)


# ---------------------------------------------------------------- classification
def classification_results(clf):
    report, matrix = clf.report()
    return {"comparison": clf.comparison_table(), "report": report, "confusion": matrix,
            "misclassified": clf.misclassified(), "best": clf.best_name}


# ---------------------------------------------------------------- retrieval
ABLATION = [
    ("Phase I baseline: TF-IDF", dict(stem=False), dict(alpha=0.0, expand=False)),
    ("+ stemming", dict(stem=True), dict(alpha=0.0, expand=False)),
    ("+ LSA semantic (hybrid, final)", dict(stem=True), dict(alpha=0.5, expand=False)),
    ("LSA semantic only", dict(stem=True), dict(alpha=1.0, expand=False)),
    ("hybrid + ontology query expansion", dict(stem=True), dict(alpha=0.5, expand=True)),
    ("TF-IDF + ontology expansion (no stemming)", dict(stem=False), dict(alpha=0.0, expand=True)),
]


def retrieval_ablation(df, queries=None):
    queries = load_queries() if queries is None else queries
    rows, per_query_final = [], None
    cache = {}
    for name, build_args, search_args in ABLATION:
        key = tuple(sorted(build_args.items()))
        if key not in cache:
            cache[key] = Recommender(df, **build_args)
        rec = cache[key]
        per_query, summary = evaluate_search(lambda q: rec.search(q, k=10, **search_args)["id"].tolist(), queries)
        rows.append({"configuration": name, **summary})
        if "final" in name:
            per_query_final = per_query
    table = pd.DataFrame(rows)
    return table, per_query_final, max_precision_at_k(queries["relevant"])


# ---------------------------------------------------------------- recommendations
def recommendation_cases(df, rec, baseline=None):
    cases = pd.read_csv(EVAL_DIR / "recommendation_cases.csv")
    cases["relevant"] = cases["expected_ids"].apply(_ids)
    baseline = baseline or Recommender(df, stem=False)

    def run(r, alpha):
        return evaluate_search(lambda seed: r.similar_to(int(seed), k=10, alpha=alpha)["id"].tolist(),
                               cases, text_col="seed_id")

    per_case, final = run(rec, None)
    _, base = run(baseline, 0.0)
    per_case["seed"] = per_case["input"].apply(lambda i: df.loc[df["id"] == int(i), "title"].iloc[0])
    summary = pd.DataFrame([{"recommender": "Phase I TF-IDF", **base},
                            {"recommender": "Phase II hybrid (final)", **final}])
    return summary, per_case, max_precision_at_k(cases["relevant"])


def recommendation_consistency(df, rec, k=5):
    """For every document as seed: share of the top-k with the same category / at least one shared concept."""
    rows = []
    for doc_id, category in zip(df["id"], df["category"]):
        res = rec.similar_to(doc_id, k=k)
        seed_concepts = set(rec.doc_concepts[rec.idx(doc_id)])
        same_cat = (res["category"] == category).mean()
        shared = np.mean([bool(seed_concepts & set(rec.doc_concepts[rec.idx(i)])) for i in res["id"]])
        rows.append({"seed": doc_id, "same_category": same_cat, "shares_concept": shared,
                     "mean_score": res["score"].mean()})
    t = pd.DataFrame(rows)
    return {"same_category@5": round(t["same_category"].mean(), 3),
            "shares_a_concept@5": round(t["shares_concept"].mean(), 3),
            "mean_similarity@5": round(t["mean_score"].mean(), 3)}


# ---------------------------------------------------------------- search context
def context_evaluation(rec):
    tests = pd.read_csv(EVAL_DIR / "context_tests.csv")
    rows = []
    for _, t in tests.iterrows():
        relevant = _ids(t["relevant_ids"])
        ctx = SearchContext()
        kind, value = t["history"].split(":", 1)
        ctx.add_view(int(value)) if kind == "view" else ctx.add_query(value)

        without = rec.search(t["query"], k=10)
        without = without["id"].tolist() if not without.empty else []
        with_ctx = rec.search(t["query"], k=10, context=ctx)["id"].tolist()
        a, b = retrieval_metrics(without, relevant), retrieval_metrics(with_ctx, relevant)
        rows.append({"scenario": t["scenario_id"], "previous activity": t["history"], "query": t["query"],
                     "relevant": ", ".join(map(str, sorted(relevant))),
                     "top5 without history": ", ".join(map(str, without[:5])) or "(no results)",
                     "top5 with history": ", ".join(map(str, with_ctx[:5])),
                     "MRR without": round(a["MRR"], 3), "MRR with": round(b["MRR"], 3),
                     "R@5 without": round(a["R@5"], 3), "R@5 with": round(b["R@5"], 3)})
    t = pd.DataFrame(rows)
    summary = {"MRR without history": round(t["MRR without"].mean(), 3),
               "MRR with history": round(t["MRR with"].mean(), 3),
               "R@5 without history": round(t["R@5 without"].mean(), 3),
               "R@5 with history": round(t["R@5 with"].mean(), 3)}
    return summary, t


# ---------------------------------------------------------------- chatbot
def chatbot_evaluation(bot):
    tests = pd.read_csv(EVAL_DIR / "chatbot_tests.csv")
    rows = []
    for _, t in tests.iterrows():
        expected = _ids(t["expected_ids"])
        keyword = "" if pd.isna(t["expected_keyword"]) else str(t["expected_keyword"])
        r = bot.answer(t["question"])
        ids = r["sources"]["id"].tolist() if not r["sources"].empty else []
        in_scope = bool(expected)
        rows.append({
            "test": t["test_id"], "question": t["question"], "in_scope": in_scope,
            "expected": ", ".join(map(str, sorted(expected))) or "(none)",
            "answered": r["found"], "top3 sources": ", ".join(map(str, ids[:3])) or "-",
            "source hit@1": (bool(ids) and ids[0] in expected) if in_scope else None,
            "source hit@3": bool(set(ids[:3]) & expected) if in_scope else None,
            "answer has key fact": (keyword.lower() in r["answer"].lower()) if keyword else None,
            "correct behaviour": (r["found"] and bool(set(ids[:3]) & expected)) if in_scope else (not r["found"]),
        })
    t = pd.DataFrame(rows)
    ins, outs = t[t["in_scope"]], t[~t["in_scope"]]
    summary = {
        "in-scope questions": len(ins),
        "source hit@1": round(ins["source hit@1"].mean(), 3),
        "source hit@3": round(ins["source hit@3"].mean(), 3),
        "answer contains key fact": round(ins["answer has key fact"].mean(), 3),
        "out-of-scope questions": len(outs),
        "out-of-scope correctly rejected": round(outs["correct behaviour"].mean(), 3),
    }
    return summary, t
