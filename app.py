"""
PolloSoft IT Knowledge Hub - Phase II integrated prototype.

Run locally:  streamlit run app.py
"""
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import evaluation as ev
from src.chatbot import Chatbot
from src.classifier import DocumentClassifier
from src.knowledge_graph import (DOC_COLORS, EDGE_STYLE, NODE_STYLE, build_graph, concept_details,
                                 graph_stats, node_id, subgraph, to_html, top_concepts)
from src.ontology import CLASSES, RELATIONS, detect_concepts
from src.recommender import Recommender, SearchContext
from src.text import clean_text, load_corpus

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

st.set_page_config(page_title="PolloSoft IT Knowledge Hub", page_icon="📚", layout="wide")


# ------------------------------------------------------------------ system
@st.cache_resource(show_spinner="Loading the knowledge repository...")
def load_system():
    df = load_corpus(ROOT / "data" / "corpus.csv")
    rec = Recommender(df)
    clf = DocumentClassifier(df)
    bot = Chatbot(rec, clf)
    graph = build_graph(df, rec)
    return df, rec, clf, bot, graph


@st.cache_resource(show_spinner="Running the evaluation...")
def load_evaluation():
    df, rec, clf, bot, _ = load_system()
    return {
        "classification": ev.classification_results(clf),
        "retrieval": ev.retrieval_ablation(df),
        "recommendation": ev.recommendation_cases(df, rec),
        "consistency": (ev.recommendation_consistency(df, rec),
                        ev.recommendation_consistency(df, Recommender(df, stem=False, alpha=0.0))),
        "context": ev.context_evaluation(rec),
        "chatbot": ev.chatbot_evaluation(bot),
    }


df, rec, clf, bot, G = load_system()
titles = dict(zip(df["id"], df["title"]))

if "ctx" not in st.session_state:
    st.session_state.ctx = SearchContext()
if "chat" not in st.session_state:
    st.session_state.chat = []


def active_context():
    ctx = st.session_state.ctx
    if st.session_state.get("use_history", True) and not ctx.is_empty():
        return ctx
    return None


def score_columns():
    return {
        "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f"),
        "title": st.column_config.TextColumn("title", width="medium"),
    }


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📚 IT Knowledge Hub")
    st.caption("PolloSoft · Knowledge Management System · Phase II")
    st.toggle("Personalize results with my search history", value=True, key="use_history",
              help="Your previous searches, opened documents and chatbot questions are used to "
                   "re-rank relevant results (up to 30% of the score).")

    st.markdown("**Your search history**")
    events = st.session_state.ctx.events
    if not events:
        st.caption("Empty. Search, open documents or ask the chatbot and it will show up here.")
    else:
        icons = {"search": "🔎", "chat": "💬", "view": "📄"}
        for e in reversed(events[-8:]):
            label = titles.get(e.get("doc_id"), "") if e["type"] == "view" else e["text"]
            st.markdown(f"{icons.get(e['type'], '•')} {label}")
        if st.button("Clear history", width="stretch"):
            st.session_state.ctx.clear()
            st.session_state.chat = []
            st.session_state.pop("rec_view", None)
            st.rerun()

    st.divider()
    counts = df["category"].value_counts()
    st.caption(f"Repository: {len(df)} documents · "
               f"{counts.get('Policy/Process', 0)} policies/processes · "
               f"{counts.get('Support/Incidents', 0)} support tickets")


tab_home, tab_clf, tab_rec, tab_bot, tab_graph, tab_eval = st.tabs(
    ["Overview", "Classify", "Recommend", "Chatbot", "Knowledge Graph", "Evaluation"])

# ------------------------------------------------------------------ overview
with tab_home:
    st.header("PolloSoft IT Knowledge Hub")
    st.write("One place to find IT policies, procedures and solved support tickets. The system "
             "classifies documents, recommends related ones, answers questions in natural language "
             "and shows how everything is connected in a knowledge graph.")

    nodes, edges = graph_stats(G)
    c = st.columns(5)
    c[0].metric("Documents", len(df))
    c[1].metric("Concepts", nodes.get("concept", 0))
    c[2].metric("Graph nodes", sum(nodes.values()))
    c[3].metric("Graph edges", sum(edges.values()))
    c[4].metric("Classifier F1 (5-fold CV)", f"{clf.macro_f1():.2f}")

    st.subheader("Architecture")
    st.image(str(ROOT / "docs" / "architecture.png"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("""
**How the components share information**
- **Repository → processing:** every document is cleaned, stemmed and turned into TF-IDF vectors and LSA embeddings.
- **Recommender → chatbot:** the chatbot does not search on its own; it asks the recommender for the most relevant documents.
- **Classifier → chatbot:** the classifier predicts if a question is about a policy or an incident, and those documents get a small boost.
- **Recommender → knowledge graph:** the similarity matrix creates the *similar_to* and *governed_by* edges.
- **Interface → recommender:** every search, opened document and chatbot question goes into the session history, which re-ranks later results.
""")
    with right:
        st.markdown("""
**Try it**
1. **Recommend:** search *my VPN drops when I work from home*.
2. **Recommend:** now search *connection problem*. With your history on, VPN documents come first.
3. **Chatbot:** ask *What should I do if I lose my company phone?*
4. **Knowledge Graph:** focus on the concept *VPN* to see the documents around it.
5. **Classify:** paste a new ticket and see where it would go.
""")

# ------------------------------------------------------------------ classify
with tab_clf:
    st.header("Automated document classification")
    st.caption(f"Model: {clf.best_name} over TF-IDF, chosen by 5-fold cross-validation. "
               "Paste a new document to classify it and see where it fits in the repository.")

    examples = {
        "Write your own": ("", ""),
        "Example: new support ticket": (
            "Scanner Not Sending Emails",
            "The user reports that the scanner on the third floor does not send scanned documents to "
            "email. The SMTP settings were updated and the test scan arrived correctly."),
        "Example: new policy": (
            "USB Storage Policy",
            "The use of personal USB drives is prohibited on company computers. Employees must use the "
            "corporate cloud to transfer files, and IT will block unknown storage devices."),
    }
    choice = st.selectbox("Start from", list(examples))
    with st.form("classify_form"):
        new_title = st.text_input("Title", value=examples[choice][0], key=f"t_{choice}")
        new_text = st.text_area("Text", value=examples[choice][1], height=110, key=f"x_{choice}")
        go = st.form_submit_button("Classify document", type="primary")

    if go:
        full = f"{new_title}. {new_text}".strip(". ")
        if len(clean_text(full)) < 10:
            st.warning("Write a title and a short description first.")
        else:
            p = clf.predict(full)
            concepts = detect_concepts(clean_text(full))
            similar = rec.search(full, k=5)

            a, b = st.columns([1, 2])
            with a:
                st.metric("Predicted category", p["category"], f"{p['confidence']:.0%} confidence",
                          delta_color="off")
                for label, prob in p["probabilities"].items():
                    st.progress(prob, text=f"{label}: {prob:.0%}")
                st.markdown("**Words that pushed the decision:** " + ", ".join(f"`{w}`" for w in p["evidence"]))
            with b:
                st.markdown("**In the knowledge graph this document would connect to:**")
                lines = [f"- *belongs_to* → **{p['category']}**"]
                if concepts:
                    lines.append("- *mentions* → " + ", ".join(f"**{c}**" for c in concepts))
                if not similar.empty:
                    if p["category"] == "Support/Incidents":
                        policies = similar[similar["category"] == "Policy/Process"]
                        if not policies.empty and policies["score"].iloc[0] >= 0.3:
                            lines.append(f"- *governed_by* → **{policies['title'].iloc[0]}** (doc {policies['id'].iloc[0]})")
                    close = similar[similar["score"] >= 0.3]
                    if not close.empty:
                        lines.append("- *similar_to* → " + ", ".join(f"{t} (doc {i})" for i, t in zip(close["id"], close["title"])))
                st.markdown("\n".join(lines))

            st.markdown("**Most similar existing documents**")
            st.dataframe(similar[["rank", "id", "title", "category", "score", "shared_concepts"]],
                         hide_index=True, width="stretch", column_config=score_columns())

    with st.expander("Model details: how the classifier was chosen and how well it works"):
        st.markdown("Every model is tested with **stratified 5-fold cross-validation**, so each document is "
                    "predicted by a model that never saw it. Stop words are kept on purpose: words like "
                    "*must*, *was* and *the user* are what separate a policy from a support ticket.")
        st.dataframe(clf.comparison_table(), hide_index=True, width="stretch")
        report, matrix = clf.report()
        x, y = st.columns(2)
        x.markdown("**Per-class precision / recall**")
        x.dataframe(report, hide_index=True, width="stretch")
        y.markdown("**Confusion matrix**")
        y.dataframe(matrix, width="stretch")
        st.markdown("**Documents the model gets wrong**")
        st.dataframe(clf.misclassified(), hide_index=True, width="stretch")

# ------------------------------------------------------------------ recommend
with tab_rec:
    st.header("Document recommendations")
    st.caption("Similarity = 50% semantic (LSA embeddings) + 50% lexical (TF-IDF), both with cosine similarity. "
               "With history on: score = 70% similarity + 30% closeness to your recent activity, and that 30% is "
               "scaled by the document's own similarity, so history re-ranks relevant documents but never pushes "
               "unrelated ones up.")

    mode = st.radio("Search by", ["Free text", "Similar to a document"], horizontal=True)
    with st.form("rec_form"):
        if mode == "Free text":
            query = st.text_input("What are you looking for?", placeholder="e.g. my VPN drops when I work from home")
        else:
            seed = st.selectbox("Pick a document", df["id"].tolist(), format_func=lambda i: f"{i}. {titles[i]}")
        k = st.slider("Number of recommendations", 3, 10, 5)
        submitted = st.form_submit_button("Recommend", type="primary")

    if submitted:
        context = active_context()  # history BEFORE this search
        if mode == "Free text":
            if not query.strip():
                st.warning("Type something to search.")
            else:
                res = rec.search(query, k=k, context=context)
                st.session_state.ctx.add_query(query)
                st.session_state.rec_view = {"label": f'Query: "{query}"', "results": res,
                                             "history": context is not None}
        else:
            res = rec.similar_to(seed, k=k, context=context)
            st.session_state.ctx.add_view(seed)
            st.session_state.rec_view = {"label": f"Seed: {seed}. {titles[seed]} (excluded from results)",
                                         "results": res, "history": context is not None}
        st.rerun()

    view = st.session_state.get("rec_view")
    if view:
        res = view["results"]
        st.markdown(f"**{view['label']}**  ·  history used: {'yes' if view['history'] else 'no'}")
        if res.empty:
            st.info("None of those words are in the repository. Try other words (the repository is in English).")
        else:
            cols = ["rank", "id", "title", "category", "score", "similarity", "semantic", "lexical"]
            if view["history"]:
                cols.append("context")
            cols += ["shared_concepts", "key_terms"]
            st.dataframe(res[cols], hide_index=True, width="stretch", column_config=score_columns())
            st.caption("semantic / lexical = cosine similarity of each representation · context = closeness to "
                       "your history · shared concepts and key terms explain why the document was recommended.")
            for _, r in res.iterrows():
                with st.expander(f"{r['rank']}. {r['title']}  ·  {r['category']}  ·  score {r['score']:.3f}"):
                    st.write(df.loc[df["id"] == r["id"], "text"].iloc[0])
                    if st.button("Mark as read (adds it to my history)", key=f"read_{r['id']}"):
                        st.session_state.ctx.add_view(r["id"])
                        st.toast(f"Added to your history: {r['title']}")
                        st.rerun()

    if not st.session_state.ctx.is_empty():
        st.subheader("Recommended for you")
        st.caption("Based only on your recent activity (documents you already opened are skipped).")
        st.dataframe(rec.for_you(st.session_state.ctx), hide_index=True, width="stretch",
                     column_config=score_columns())

# ------------------------------------------------------------------ chatbot
with tab_bot:
    st.header("Ask the knowledge base")
    st.caption("The chatbot retrieves the most relevant documents, builds the answer from their text and "
               "cites them. If the repository has nothing on the topic, it says so.")

    sample = ["How often do I have to change my password?", "My VPN keeps disconnecting when I work from home",
              "What should I do if I lose my company phone?", "Can I use my personal Dropbox for work files?"]
    cols = st.columns(len(sample))
    for col, q in zip(cols, sample):
        if col.button(q, width="stretch"):
            st.session_state.pending_question = q

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                meta = f"Detected intent: {msg['intent']}"
                if msg.get("history"):
                    meta += " · used your search history"
                st.caption(meta)
                if msg.get("sources") is not None and not msg["sources"].empty:
                    with st.expander("Sources retrieved from the repository"):
                        st.dataframe(msg["sources"], hide_index=True, width="stretch",
                                     column_config=score_columns())

    question = st.chat_input("Ask about IT policies, procedures or past tickets...")
    question = question or st.session_state.pop("pending_question", None)
    if question:
        context = active_context()
        result = bot.answer(question, context=context)
        st.session_state.ctx.add_query(question, source="chat")
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": result["answer"], "intent": result["intent"],
                                      "sources": result["sources"],
                                      "history": context is not None and result["found"]})
        st.rerun()

# ------------------------------------------------------------------ knowledge graph
with tab_graph:
    st.header("Knowledge graph")
    st.caption("Documents, concepts, domains and categories from the ontology, and how they relate. "
               "Drag nodes, zoom, and hover to read a document.")

    labels = {n: (d["label"] if d["kind"] != "document" else f"{d['doc_id']}. {d['label']}")
              for n, d in G.nodes(data=True)}
    kinds = ["category", "domain", "concept", "document"]

    c1, c2, c3 = st.columns([1.1, 1.3, 1.6])
    with c1:
        view_mode = st.radio("View", ["Whole graph", "Focus on a node"], horizontal=True)
        focus, radius = None, 1
        if view_mode == "Focus on a node":
            options = sorted([n for n in G if G.nodes[n]["kind"] == "concept"], key=labels.get) + \
                      sorted([n for n in G if G.nodes[n]["kind"] != "concept"], key=labels.get)
            default = options.index(node_id("concept", "VPN")) if node_id("concept", "VPN") in options else 0
            focus = st.selectbox("Node", options, index=default, format_func=lambda n: labels[n])
            radius = st.slider("Distance (hops)", 1, 2, 1)
    with c2:
        node_types = st.multiselect("Node types", kinds, default=kinds)
        categories = st.multiselect("Document categories", sorted(DOC_COLORS), default=sorted(DOC_COLORS))
    with c3:
        relations = st.multiselect("Relations", list(EDGE_STYLE), default=list(EDGE_STYLE))
        highlight_history = st.checkbox("Highlight documents from my history", value=True)

    H = subgraph(G, node_types=node_types, relations=relations, categories=categories, focus=focus, radius=radius)
    highlight = []
    if highlight_history:
        highlight = [node_id("doc", e["doc_id"]) for e in st.session_state.ctx.events if e["type"] == "view"]

    legend = " ".join(
        f"<span style='display:inline-block;margin-right:14px'><span style='display:inline-block;width:12px;height:12px;"
        f"border-radius:50%;background:{color};margin-right:5px;vertical-align:middle'></span>{name}</span>"
        for name, color in [("Category", NODE_STYLE["category"]["color"]), ("Domain", NODE_STYLE["domain"]["color"]),
                            ("Concept", NODE_STYLE["concept"]["color"]), ("Policy/Process doc", DOC_COLORS["Policy/Process"]),
                            ("Support ticket", DOC_COLORS["Support/Incidents"])])
    legend += "<br>" + " ".join(
        f"<span style='margin-right:14px;color:{s['color']}'>{'┄┄' if s['dashes'] else '──'} {r}</span>"
        for r, s in EDGE_STYLE.items())
    st.markdown(f"<div style='font-size:0.85rem'>{legend}</div>", unsafe_allow_html=True)

    if H.number_of_nodes() == 0:
        st.info("Nothing to show with these filters.")
    else:
        st.caption(f"Showing {H.number_of_nodes()} nodes and {H.number_of_edges()} edges.")
        short = H.number_of_nodes() > 40
        if short:
            st.caption("Big view: documents show only their id. Hover a node to see its title and text, "
                       "or use *Focus on a node* for a closer look.")
        components.html(to_html(H, highlight=highlight, focus=focus, short_doc_labels=short),
                        height=660, scrolling=False)

    left, right = st.columns(2)
    with left:
        st.subheader("Explore a concept")
        concept_names = sorted(d["label"] for _, d in G.nodes(data=True) if d["kind"] == "concept")
        chosen = st.selectbox("Concept", concept_names, index=concept_names.index("VPN") if "VPN" in concept_names else 0)
        docs, related = concept_details(G, chosen)
        st.markdown(f"**Domain:** {G.nodes[node_id('concept', chosen)]['domain']} · **mentioned by {len(docs)} documents**")
        st.dataframe(pd.DataFrame(docs), hide_index=True, width="stretch")
        if related:
            st.markdown("**Often appears with:** " + ", ".join(f"{r['concept']} ({r['shared_documents']} docs)" for r in related))
    with right:
        st.subheader("Graph summary")
        nodes, edges = graph_stats(G)
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame(nodes.items(), columns=["node type", "count"]), hide_index=True, width="stretch")
        b.dataframe(pd.DataFrame(edges.items(), columns=["relation", "count"]), hide_index=True, width="stretch")
        st.markdown("**Most connected concepts**")
        st.dataframe(pd.DataFrame(top_concepts(G, 8)), hide_index=True, width="stretch")

    with st.expander("Ontology schema (classes and relations)"):
        st.dataframe(pd.DataFrame([{"class": k, "description": v} for k, v in CLASSES.items()]),
                     hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([{"relation": r, "from": a, "to": b, "meaning": m}
                                   for r, (a, b, m) in RELATIONS.items()]),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.header("System evaluation")
    st.caption("All numbers are computed live from the hand-labeled test sets in the eval/ folder.")
    E = load_evaluation()

    st.subheader("1. Classification: precision and recall")
    cls = E["classification"]
    st.markdown("Stratified 5-fold cross-validation over the 60 documents (out-of-fold predictions).")
    st.dataframe(cls["comparison"], hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.dataframe(cls["report"], hide_index=True, width="stretch")
    b.dataframe(cls["confusion"], width="stretch")

    st.subheader("2. Retrieval: precision and recall of free-text search")
    table, per_query, ceiling = E["retrieval"]
    st.markdown(f"25 paraphrased queries with hand-labeled relevant documents (1–4 each). "
                f"Because most queries have fewer than 5 relevant documents, the best possible P@5 is **{ceiling}**.")
    st.dataframe(table, hide_index=True, width="stretch")
    with st.expander("Per-query results (final configuration)"):
        st.dataframe(per_query, hide_index=True, width="stretch")

    st.subheader("3. Recommendation quality: test cases")
    summary, per_case, rec_ceiling = E["recommendation"]
    st.markdown(f"10 seed documents with the documents a user would expect to see next (P@5 ceiling {rec_ceiling}).")
    st.dataframe(summary, hide_index=True, width="stretch")
    with st.expander("Per-case results"):
        st.dataframe(per_case, hide_index=True, width="stretch")
    final_c, base_c = E["consistency"]
    st.dataframe(pd.DataFrame([{"recommender": "Phase I TF-IDF", **base_c},
                               {"recommender": "Phase II hybrid (final)", **final_c}]),
                 hide_index=True, width="stretch")
    st.caption("Using every document as a seed. Same-category share goes down on purpose: the hybrid model links "
               "support tickets with the policy that covers them.")

    st.subheader("4. Search history: does context help?")
    ctx_summary, ctx_table = E["context"]
    st.markdown("6 ambiguous queries (e.g. *connection problem*) run with and without one previous search.")
    st.dataframe(pd.DataFrame([ctx_summary]), hide_index=True, width="stretch")
    with st.expander("Scenarios"):
        st.dataframe(ctx_table, hide_index=True, width="stretch")

    st.subheader("5. Chatbot")
    bot_summary, bot_table = E["chatbot"]
    st.dataframe(pd.DataFrame([bot_summary]), hide_index=True, width="stretch")
    with st.expander("All chatbot test questions"):
        st.dataframe(bot_table, hide_index=True, width="stretch")

    st.subheader("6. Usability")
    usability = ROOT / "docs" / "usability.md"
    if usability.exists():
        st.markdown(usability.read_text())
