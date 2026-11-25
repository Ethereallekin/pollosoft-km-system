"""
Runs the full evaluation and saves the results in eval/results/.

    python eval/run_evaluation.py
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import evaluation as ev  # noqa: E402
from src.chatbot import Chatbot  # noqa: E402
from src.classifier import DocumentClassifier  # noqa: E402
from src.knowledge_graph import build_graph, graph_stats  # noqa: E402
from src.recommender import Recommender  # noqa: E402
from src.text import load_corpus  # noqa: E402

OUT = ROOT / "eval" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = load_corpus(ROOT / "data" / "corpus.csv")
    rec = Recommender(df)
    clf = DocumentClassifier(df)
    bot = Chatbot(rec, clf)
    G = build_graph(df, rec)

    cls = ev.classification_results(clf)
    cls["comparison"].to_csv(OUT / "classification_models.csv", index=False)
    cls["report"].to_csv(OUT / "classification_report.csv", index=False)
    cls["confusion"].to_csv(OUT / "classification_confusion.csv")
    cls["misclassified"].to_csv(OUT / "classification_errors.csv", index=False)

    ablation, per_query, ceiling = ev.retrieval_ablation(df)
    ablation.to_csv(OUT / "retrieval_ablation.csv", index=False)
    per_query.to_csv(OUT / "retrieval_per_query.csv", index=False)

    rec_summary, rec_cases, rec_ceiling = ev.recommendation_cases(df, rec)
    rec_summary.to_csv(OUT / "recommendation_summary.csv", index=False)
    rec_cases.to_csv(OUT / "recommendation_cases.csv", index=False)
    consistency = ev.recommendation_consistency(df, rec)
    baseline_consistency = ev.recommendation_consistency(df, Recommender(df, stem=False, alpha=0.0))

    ctx_summary, ctx_table = ev.context_evaluation(rec)
    ctx_table.to_csv(OUT / "context_history.csv", index=False)

    bot_summary, bot_table = ev.chatbot_evaluation(bot)
    bot_table.to_csv(OUT / "chatbot_tests.csv", index=False)

    nodes, edges = graph_stats(G)
    best_row = cls["comparison"].iloc[0].to_dict()
    metrics = {
        "corpus": {"documents": len(df), "by_category": df["category"].value_counts().to_dict()},
        "classification": {"best_model": cls["best"], **{k: v for k, v in best_row.items() if k != "model"},
                           "per_class": cls["report"].to_dict(orient="records")},
        "retrieval": {"queries": len(per_query), "P@5_ceiling": ceiling,
                      "ablation": ablation.to_dict(orient="records")},
        "recommendation": {"cases": len(rec_cases), "P@5_ceiling": rec_ceiling,
                           "summary": rec_summary.to_dict(orient="records"),
                           "consistency_final": consistency, "consistency_phase1": baseline_consistency},
        "search_context": ctx_summary,
        "chatbot": bot_summary,
        "knowledge_graph": {"nodes": nodes, "edges": edges},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print("== Classification (5-fold CV) ==")
    print(cls["comparison"].to_string(index=False))
    print(cls["report"].to_string(index=False))
    print("\n== Retrieval ablation (25 queries) == P@5 ceiling", ceiling)
    print(ablation.to_string(index=False))
    print("\n== Recommendation cases (10 seeds) == P@5 ceiling", rec_ceiling)
    print(rec_summary.to_string(index=False))
    print("consistency final:", consistency)
    print("consistency phase1:", baseline_consistency)
    print("\n== Search context ==", ctx_summary)
    print("\n== Chatbot ==", bot_summary)
    print("\n== Graph ==", nodes, edges)
    print(f"\nResults saved in {OUT}")


if __name__ == "__main__":
    main()
