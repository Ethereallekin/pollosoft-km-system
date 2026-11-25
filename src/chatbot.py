"""
Chatbot for natural language queries (retrieval-based answer generation).

Flow for every question:
  1. rules: greetings, thanks and help get a fixed reply
  2. retrieval: the recommender finds the most relevant documents
     (stemmed TF-IDF + LSA, plus the session context for follow-up questions)
  3. the classifier predicts if the question is about a policy or an incident,
     and documents of that category get a small boost
  4. generation: the answer is built from the retrieved text: the full main
     document plus the most relevant sentence of a supporting one (a policy for
     an incident or a past ticket for a policy question), citing both
  5. out-of-scope check: if most words of the question are unknown to the
     repository, or no document is similar enough, the bot says so
"""
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .text import clean_text, split_sentences

GREETINGS = {"hi", "hello", "hey", "hola", "good morning", "good afternoon", "good evening"}
THANKS = {"thanks", "thank you", "thx", "gracias", "great thanks"}
HELP = {"help", "what can you do", "how do you work", "what can i ask"}

HELP_TEXT = (
    "I answer questions using the IT knowledge repository (policies, procedures and resolved "
    "support tickets). Try things like:\n"
    "- *How often do I have to change my password?*\n"
    "- *My VPN keeps disconnecting when I work from home*\n"
    "- *What should I do if I lose my company phone?*"
)


class Chatbot:
    def __init__(self, recommender, classifier, min_score=0.3, min_coverage=0.5, category_boost=0.05):
        self.rec = recommender
        self.clf = classifier
        self.min_score = min_score
        self.min_coverage = min_coverage
        self.category_boost = category_boost

        rows = []
        for _, doc in recommender.df.iterrows():
            for sentence in split_sentences(doc["text"]):
                rows.append({"doc_id": doc["id"], "title": doc["title"],
                             "category": doc["category"], "sentence": sentence})
        self.sentences = pd.DataFrame(rows)
        self.sentence_vecs = recommender.vectorizer.transform(self.sentences["sentence"].apply(clean_text))

    # ---------- helpers ----------
    @staticmethod
    def _small_talk(question):
        q = re.sub(r"[^\w\s]", "", question.lower()).strip()
        if q in GREETINGS:
            return "Hi! Ask me anything about IT policies, procedures or past support tickets."
        if q in THANKS:
            return "You're welcome! Let me know if you need anything else."
        if q in HELP:
            return HELP_TEXT
        return None

    def _pick_sentences(self, question, doc_id, all_sentences):
        """All sentences of the main document, or the single best one of a supporting document."""
        mask = (self.sentences["doc_id"] == doc_id).values
        doc_sents = self.sentences[mask].reset_index(drop=True)
        if all_sentences or len(doc_sents) == 1:
            return doc_sents["sentence"].tolist()
        q = self.rec.vectorizer.transform([clean_text(question)])
        scores = cosine_similarity(q, self.sentence_vecs[mask]).flatten()
        return [doc_sents.loc[int(np.argmax(scores)), "sentence"]]

    # ---------- main ----------
    def answer(self, question, context=None, k_docs=3):
        canned = self._small_talk(question)
        if canned:
            return {"answer": canned, "sources": pd.DataFrame(), "intent": "small talk",
                    "predicted_category": None, "found": True}

        prediction = self.clf.predict(question)
        # out-of-scope check (history never makes an off-topic question valid)
        if self.rec.coverage(question) < self.min_coverage:
            return self._not_found(prediction)
        results = self.rec.search(question, k=10, context=context, beta=0.2)
        if results.empty:
            return self._not_found(prediction)

        # boost documents from the category the question seems to be about
        if prediction["confidence"] >= 0.6:
            boost = np.where(results["category"] == prediction["category"], self.category_boost, 0)
            results["score"] = (results["score"] + boost).round(4)
            results = results.sort_values("score", ascending=False).reset_index(drop=True)
            results["rank"] = range(1, len(results) + 1)

        top = results[results["score"] >= self.min_score].head(k_docs)
        if top.empty:
            return self._not_found(prediction)

        # main document in full + one supporting document: preferably of the other type
        # (a policy for an incident, a solved ticket for a policy question), otherwise the
        # next one if its score is close enough
        main = top.iloc[[0]]
        rest = top.iloc[1:]
        other_type = rest[rest["category"] != main["category"].iloc[0]]
        if not other_type.empty:
            support = other_type.head(1)
        else:
            support = rest[rest["score"] >= main["score"].iloc[0] * 0.75].head(1)
        used = pd.concat([main, support])

        intent = "incident" if prediction["category"] == "Support/Incidents" else "policy / how-to"
        top_is_ticket = used["category"].iloc[0] == "Support/Incidents"
        opener = ("A similar case was already solved. Here's what worked:" if top_is_ticket
                  else "Here's what our policies and procedures say:")
        lines = [opener]
        for n, (doc_id, title, category) in enumerate(zip(used["id"], used["title"], used["category"])):
            label = "past ticket" if category == "Support/Incidents" else "policy"
            lines.append(f"\n**{title}** ({label}, doc {doc_id})")
            for sentence in self._pick_sentences(question, doc_id, all_sentences=(n == 0)):
                lines.append(f"- {sentence}")

        others = top[~top["id"].isin(used["id"])]
        if not others.empty:
            lines.append("\nRelated: " + ", ".join(f"{t} (doc {i})" for i, t in zip(others["id"], others["title"])))

        sources = top[["rank", "id", "title", "category", "score", "shared_concepts"]].copy()
        sources["used_in_answer"] = sources["id"].isin(used["id"])
        return {"answer": "\n".join(lines), "sources": sources, "intent": intent,
                "predicted_category": prediction["category"], "found": True}

    @staticmethod
    def _not_found(prediction):
        return {"answer": ("I couldn't find anything about that in the knowledge repository. "
                           "Try rephrasing with IT terms (e.g. password, VPN, printer, email), "
                           "or open a help desk ticket."),
                "sources": pd.DataFrame(), "intent": "out of scope",
                "predicted_category": prediction["category"], "found": False}
