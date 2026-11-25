"""
Automated document classification (refined version).

Phase I sorted documents by hand into Policy/Process and Support/Incidents.
Here a supervised model learns that split from the repository:
  - pipeline: TF-IDF (unigrams, stop words kept) -> classifier
  - model selection: several candidates compared with stratified 5-fold
    cross-validation, the best macro-F1 wins
  - output: predicted category, confidence and the words that pushed the decision
A keyword rule baseline is kept to show how much the learned model improves.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .text import clean_text

POLICY_WORDS = ["policy", "procedure", "process", "guide", "must", "mandatory", "required",
                "prohibited", "rules", "steps", "instructions", "plan"]


# articles and prepositions carry no meaning when shown as evidence to the user
TRIVIAL = {"the", "a", "an", "and", "of", "to", "in", "on", "at", "for", "that", "this", "it", "is", "by", "with"}


def keyword_baseline(clean):
    # simple rule: policy-like words -> Policy/Process, otherwise Support/Incidents
    words = set(clean.split())
    return "Policy/Process" if words & set(POLICY_WORDS) else "Support/Incidents"


def _make(model, stop_words=None):
    # stop words are KEPT on purpose: "must", "was", "the user" are what separate a
    # prescriptive policy from a past-tense support ticket
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=stop_words, sublinear_tf=True)),
        ("model", model),
    ])


CANDIDATES = {
    "Logistic Regression": lambda: _make(LogisticRegression(C=10, max_iter=3000)),
    "Multinomial Naive Bayes": lambda: _make(MultinomialNB(alpha=0.3)),
    "Logistic Regression (stop words removed)": lambda: _make(LogisticRegression(C=10, max_iter=3000), "english"),
}


class DocumentClassifier:
    def __init__(self, df, folds=5, seed=42):
        self.df = df
        self.labels = sorted(df["category"].unique())
        self.cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        X, y = df["clean_text"], df["category"]
        self.comparison = []
        self.cv_predictions = {}
        for name, build in CANDIDATES.items():
            pred = cross_val_predict(build(), X, y, cv=self.cv)
            self.cv_predictions[name] = pred
            self.comparison.append(self._scores(name, y, pred))

        baseline = X.apply(keyword_baseline)
        self.cv_predictions["Keyword rules (baseline)"] = baseline.values
        self.comparison.append(self._scores("Keyword rules (baseline)", y, baseline))

        learned = [r for r in self.comparison if r["model"] in CANDIDATES]
        self.best_name = max(learned, key=lambda r: r["macro_f1"])["model"]
        self.model = CANDIDATES[self.best_name]().fit(X, y)

    @staticmethod
    def _scores(name, y, pred):
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        return {"model": name, "accuracy": round(accuracy_score(y, pred), 3),
                "macro_precision": round(p, 3), "macro_recall": round(r, 3), "macro_f1": round(f, 3)}

    def comparison_table(self):
        return pd.DataFrame(self.comparison).sort_values("macro_f1", ascending=False).reset_index(drop=True)

    def report(self, name=None):
        """Per-class precision / recall / F1 from cross-validation (out-of-fold predictions)."""
        name = name or self.best_name
        y, pred = self.df["category"], self.cv_predictions[name]
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        rows = [{"class": c, "precision": round(rep[c]["precision"], 3), "recall": round(rep[c]["recall"], 3),
                 "f1": round(rep[c]["f1-score"], 3), "support": int(rep[c]["support"])} for c in self.labels]
        matrix = pd.DataFrame(confusion_matrix(y, pred, labels=self.labels),
                              index=[f"true: {c}" for c in self.labels],
                              columns=[f"pred: {c}" for c in self.labels])
        return pd.DataFrame(rows), matrix

    def misclassified(self, name=None):
        name = name or self.best_name
        pred = self.cv_predictions[name]
        out = self.df[["id", "title", "category"]].copy()
        out["predicted"] = pred
        return out[out["category"] != out["predicted"]].reset_index(drop=True)

    def predict(self, text, top_n=6):
        clean = clean_text(text)
        proba = self.model.predict_proba([clean])[0]
        classes = list(self.model.classes_)
        best = int(np.argmax(proba))
        return {
            "category": classes[best],
            "confidence": round(float(proba[best]), 3),
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
            "evidence": self._evidence(clean, classes[best], top_n),
        }

    def _evidence(self, clean, label, top_n):
        # words of the text that pushed the prediction toward the chosen class
        tfidf = self.model.named_steps["tfidf"]
        model = self.model.named_steps["model"]
        x = tfidf.transform([clean]).toarray()[0]
        terms = tfidf.get_feature_names_out()
        classes = list(model.classes_)
        if hasattr(model, "coef_"):
            weights = model.coef_[0] if label == classes[1] else -model.coef_[0]
        else:  # naive bayes
            i = classes.index(label)
            weights = model.feature_log_prob_[i] - model.feature_log_prob_[1 - i]
        contrib = x * weights
        best = [j for j in np.argsort(-contrib) if contrib[j] > 0 and terms[j] not in TRIVIAL]
        return [terms[j] for j in best[:top_n]]

    def macro_f1(self, name=None):
        name = name or self.best_name
        return f1_score(self.df["category"], self.cv_predictions[name], average="macro")
