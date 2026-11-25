"""
Recommendation system.

Each document is represented two ways:
  - lexical:  TF-IDF vector (exact words)
  - semantic: LSA embedding (TruncatedSVD over TF-IDF), which groups words that
              appear in similar contexts (virus ~ malware ~ antivirus)

Search score = hybrid of both cosine similarities. Words are stemmed
(connecting -> connect). Ontology-based query expansion is available but off by
default: in the evaluation it raised recall only when stemming was not used.

The search context (query history + viewed documents of the session) is turned
into a context vector. Documents close to it get a boost, but only in proportion
to their own relevance, so the history re-ranks relevant results instead of
replacing them.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import snowballstemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .ontology import detect_concepts, expand_query
from .text import clean_text


@dataclass
class SearchContext:
    """Query history of a user session (queries, viewed docs, chatbot questions)."""
    events: list = field(default_factory=list)
    max_events: int = 6
    decay: float = 0.7

    def add_query(self, text, source="search"):
        if text and text.strip():
            self.events.append({"type": source, "text": text.strip()})

    def add_view(self, doc_id):
        self.events.append({"type": "view", "doc_id": int(doc_id)})

    def clear(self):
        self.events.clear()

    def recent(self):
        return self.events[-self.max_events:]

    def is_empty(self):
        return len(self.events) == 0


def make_analyzer(stem=True):
    # default word tokenizer + English stop words, optionally stemmed (connecting -> connect)
    base = TfidfVectorizer(stop_words="english").build_analyzer()
    if not stem:
        return base
    stemmer = snowballstemmer.stemmer("english")
    return lambda text: stemmer.stemWords(base(text))


class Recommender:
    def __init__(self, df, n_components=30, alpha=0.5, expand=False, expand_weight=0.3, stem=True):
        self.df = df
        self.alpha = alpha                  # weight of semantic vs lexical similarity
        self.expand = expand                # ontology-based query expansion
        self.expand_weight = expand_weight  # weight of the added ontology terms
        self.gate = 0.3                     # similarity needed to get the full history boost

        self.vectorizer = TfidfVectorizer(analyzer=make_analyzer(stem), sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(df["clean_text"])

        n_components = min(n_components, self.tfidf.shape[1] - 1, len(df) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.lsa = normalize(self.svd.fit_transform(self.tfidf))

        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(df["id"])}
        self.doc_concepts = [detect_concepts(t) for t in df["clean_text"]]
        self.terms = self.vectorizer.get_feature_names_out()
        self.readable = self._readable_terms(df["clean_text"], stem)

    @staticmethod
    def _readable_terms(texts, stem):
        # map each stem back to its most common word, so the UI shows "policy" instead of "polici"
        base = make_analyzer(stem=False)
        stemmer = snowballstemmer.stemmer("english") if stem else None
        counts = {}
        for text in texts:
            for word in base(text):
                key = stemmer.stemWord(word) if stemmer else word
                counts.setdefault(key, {}).setdefault(word, 0)
                counts[key][word] += 1
        return {k: max(v, key=v.get) for k, v in counts.items()}

    # ---------- vectors ----------
    def embed_text(self, text, expand=None, expand_weight=None):
        expand = self.expand if expand is None else expand
        w = self.expand_weight if expand_weight is None else expand_weight
        clean = clean_text(text)
        concepts = detect_concepts(clean)
        x = self.vectorizer.transform([clean])
        if expand and concepts:
            # ontology terms count, but less than the user's own words
            expanded, _ = expand_query(clean)
            extra = expanded[len(clean):]
            x = normalize(x + w * normalize(self.vectorizer.transform([extra])))
        z = normalize(self.svd.transform(x))
        return x, z, concepts

    def embed_doc(self, doc_id):
        i = self.idx(doc_id)
        return self.tfidf[i], self.lsa[i:i + 1]

    def idx(self, doc_id):
        if int(doc_id) not in self.id_to_idx:
            raise ValueError(f"No document with id={doc_id}")
        return self.id_to_idx[int(doc_id)]

    # ---------- scores ----------
    def _similarity(self, x, z, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        lexical = cosine_similarity(x, self.tfidf).flatten()
        semantic = cosine_similarity(z, self.lsa).flatten()
        semantic = np.clip(semantic, 0, 1)
        return alpha * semantic + (1 - alpha) * lexical, lexical, semantic

    def context_vector(self, context):
        if context is None or context.is_empty():
            return None
        vectors, weights = [], []
        events = context.recent()
        for age, event in enumerate(reversed(events)):
            if event["type"] == "view":
                if event["doc_id"] not in self.id_to_idx:
                    continue
                _, z = self.embed_doc(event["doc_id"])
            else:
                _, z, _ = self.embed_text(event["text"])
            vectors.append(z[0])
            weights.append(context.decay ** age)
        if not vectors:
            return None
        v = np.average(np.array(vectors), axis=0, weights=weights)
        return normalize(v.reshape(1, -1))

    def _rank(self, base, lexical, semantic, k, exclude=(), context=None, beta=0.3,
              query_concepts=None, query_vec=None):
        ctx = self.context_vector(context)
        if ctx is not None:
            ctx_score = np.clip(cosine_similarity(ctx, self.lsa).flatten(), 0, 1)
            # history only boosts documents that are already relevant: the boost is scaled
            # by the document's own similarity (full boost from similarity >= gate)
            relevance = np.ones_like(base) if base.max() == 0 else np.clip(base / self.gate, 0, 1)
            final = (1 - beta) * base + beta * ctx_score * relevance
        else:
            ctx_score = np.zeros_like(base)
            final = base

        exclude_idx = {self.idx(d) for d in exclude if int(d) in self.id_to_idx}
        order = [i for i in np.argsort(-final) if i not in exclude_idx][:k]

        rows = []
        for rank, i in enumerate(order, start=1):
            doc_concepts = self.doc_concepts[i]
            shared = [c for c in doc_concepts if query_concepts and c in query_concepts]
            rows.append({
                "rank": rank,
                "id": int(self.df["id"].iloc[i]),
                "title": self.df["title"].iloc[i],
                "category": self.df["category"].iloc[i],
                "score": round(float(final[i]), 4),
                "similarity": round(float(base[i]), 4),
                "semantic": round(float(semantic[i]), 4),
                "lexical": round(float(lexical[i]), 4),
                "context": round(float(ctx_score[i]), 4) if ctx is not None else None,
                "shared_concepts": ", ".join(shared),
                "key_terms": ", ".join(self._shared_terms(query_vec, i)) if query_vec is not None else "",
            })
        return pd.DataFrame(rows)

    def _shared_terms(self, query_vec, i, n=4):
        q = query_vec.toarray().flatten()
        d = self.tfidf[i].toarray().flatten()
        overlap = q * d
        best = np.argsort(-overlap)[:n]
        return [self.readable.get(self.terms[j], self.terms[j]) for j in best if overlap[j] > 0]

    # ---------- public API ----------
    def search(self, query, k=5, context=None, beta=0.3, exclude=(), alpha=None, expand=None,
               expand_weight=None):
        """Top-k documents for a free-text query (used by the chatbot too)."""
        x, z, concepts = self.embed_text(query, expand=expand, expand_weight=expand_weight)
        if x.nnz == 0:
            # none of the words are in the repository: only the session context can help
            if self.context_vector(context) is None:
                return pd.DataFrame()
            zeros = np.zeros(len(self.df))
            return self._rank(zeros, zeros, zeros, k, exclude, context, beta=1.0)
        base, lexical, semantic = self._similarity(x, z, alpha)
        return self._rank(base, lexical, semantic, k, exclude, context, beta, concepts, x)

    def similar_to(self, doc_id, k=5, context=None, beta=0.3, alpha=None):
        """Top-k documents similar to a seed document. The seed is excluded."""
        x, z = self.embed_doc(doc_id)
        base, lexical, semantic = self._similarity(x, z, alpha)
        concepts = self.doc_concepts[self.idx(doc_id)]
        return self._rank(base, lexical, semantic, k, [doc_id], context, beta, concepts, x)

    def for_you(self, context, k=5):
        """Recommendations based only on the session history."""
        ctx = self.context_vector(context)
        if ctx is None:
            return pd.DataFrame()
        score = np.clip(cosine_similarity(ctx, self.lsa).flatten(), 0, 1)
        viewed = [e["doc_id"] for e in context.events if e["type"] == "view"]
        zeros = np.zeros_like(score)
        df = self._rank(score, zeros, score, k, viewed)
        return df[["rank", "id", "title", "category", "score"]]

    FILLER = {"got", "get", "need", "want", "ask", "say", "keep", "doe", "happen", "make", "know", "tell"}

    def coverage(self, text):
        """Share of the meaningful words of a text that exist in the repository vocabulary."""
        tokens = [t for t in self.vectorizer.build_analyzer()(clean_text(text)) if t not in self.FILLER]
        if not tokens:
            return 0.0
        return sum(t in self.vectorizer.vocabulary_ for t in tokens) / len(tokens)

    def similarity_matrix(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        lexical = cosine_similarity(self.tfidf)
        semantic = np.clip(cosine_similarity(self.lsa), 0, 1)
        return alpha * semantic + (1 - alpha) * lexical
