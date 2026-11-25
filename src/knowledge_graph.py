"""
Knowledge graph built from the ontology (see ontology.py).

Nodes: Category, Domain, Concept, Document
Edges: belongs_to, mentions, part_of, similar_to, governed_by, co_occurs_with
Rendered as an interactive HTML network with PyVis.
"""
from collections import Counter
from itertools import combinations

import networkx as nx
from pyvis.network import Network

from .ontology import CONCEPTS, concept_domain

NODE_STYLE = {
    "category": {"color": "#E4572E", "shape": "diamond", "size": 34},
    "domain":   {"color": "#7B2CBF", "shape": "box", "size": 22},
    "concept":  {"color": "#F2A541", "shape": "dot", "size": 16},
    "document": {"color": "#4C9BE8", "shape": "dot", "size": 11},
}
DOC_COLORS = {"Policy/Process": "#2E86AB", "Support/Incidents": "#3BB273"}
EDGE_STYLE = {
    "belongs_to":     {"color": "#E4572E", "dashes": False, "width": 1},
    "mentions":       {"color": "#F2A541", "dashes": False, "width": 1},
    "part_of":        {"color": "#7B2CBF", "dashes": False, "width": 2},
    "similar_to":     {"color": "#9AA5B1", "dashes": True,  "width": 1},
    "governed_by":    {"color": "#C2185B", "dashes": False, "width": 2},
    "co_occurs_with": {"color": "#D4A373", "dashes": True,  "width": 1},
}


def node_id(kind, key):
    return f"{kind}:{key}"


def build_graph(df, recommender, sim_threshold=0.30, governed_threshold=0.30, cooccur_min=2):
    G = nx.Graph()
    sim = recommender.similarity_matrix()

    for cat in sorted(df["category"].unique()):
        G.add_node(node_id("category", cat), label=cat, kind="category")
    for name, (domain, _) in CONCEPTS.items():
        G.add_node(node_id("domain", domain), label=domain, kind="domain")
        G.add_node(node_id("concept", name), label=name, kind="concept", domain=domain)
        G.add_edge(node_id("concept", name), node_id("domain", domain), relation="part_of")

    concept_counts = Counter()
    pair_counts = Counter()
    for i, doc in df.iterrows():
        d = node_id("doc", doc["id"])
        G.add_node(d, label=doc["title"], kind="document", category=doc["category"],
                   doc_id=int(doc["id"]), text=doc["text"])
        G.add_edge(d, node_id("category", doc["category"]), relation="belongs_to")
        concepts = recommender.doc_concepts[i]
        for c in concepts:
            G.add_edge(d, node_id("concept", c), relation="mentions")
        concept_counts.update(concepts)
        pair_counts.update(combinations(sorted(concepts), 2))

    # concept <-> concept when they are mentioned together in 2+ documents
    for (a, b), n in pair_counts.items():
        if n >= cooccur_min:
            G.add_edge(node_id("concept", a), node_id("concept", b), relation="co_occurs_with", weight=n)

    ids = df["id"].tolist()
    cats = df["category"].tolist()
    for i, j in combinations(range(len(ids)), 2):
        if sim[i, j] >= sim_threshold:
            G.add_edge(node_id("doc", ids[i]), node_id("doc", ids[j]),
                       relation="similar_to", weight=round(float(sim[i, j]), 3))

    # each support ticket -> the closest policy/procedure (if similar enough)
    policies = [i for i, c in enumerate(cats) if c == "Policy/Process"]
    for i, c in enumerate(cats):
        if c != "Support/Incidents":
            continue
        best = max(policies, key=lambda j: sim[i, j])
        if sim[i, best] >= governed_threshold:
            a, b = node_id("doc", ids[i]), node_id("doc", ids[best])
            # governed_by is more specific than similar_to, so it replaces it
            G.add_edge(a, b, relation="governed_by", weight=round(float(sim[i, best]), 3))

    # drop concepts nobody mentions (keeps the graph clean)
    for name in CONCEPTS:
        if concept_counts[name] == 0:
            G.remove_node(node_id("concept", name))
    for n in [n for n, d in G.nodes(data=True) if d["kind"] == "domain" and G.degree(n) == 0]:
        G.remove_node(n)
    return G


def graph_stats(G):
    nodes = Counter(d["kind"] for _, d in G.nodes(data=True))
    edges = Counter(d["relation"] for _, _, d in G.edges(data=True))
    return dict(nodes), dict(edges)


def top_concepts(G, n=10):
    rows = []
    for node, data in G.nodes(data=True):
        if data["kind"] != "concept":
            continue
        docs = [m for m in G.neighbors(node) if G.nodes[m]["kind"] == "document"]
        rows.append({"concept": data["label"], "domain": data["domain"], "documents": len(docs),
                     "centrality": round(nx.degree_centrality(G)[node], 3)})
    rows.sort(key=lambda r: (-r["documents"], r["concept"]))
    return rows[:n]


def concept_details(G, concept):
    node = node_id("concept", concept)
    if node not in G:
        return [], []
    docs, related = [], []
    for m in G.neighbors(node):
        data = G.nodes[m]
        if data["kind"] == "document":
            docs.append({"id": data["doc_id"], "title": data["label"], "category": data["category"]})
        elif data["kind"] == "concept":
            related.append({"concept": data["label"], "shared_documents": G.edges[node, m].get("weight", 0)})
    docs.sort(key=lambda r: r["id"])
    related.sort(key=lambda r: -r["shared_documents"])
    return docs, related


def subgraph(G, node_types=None, relations=None, categories=None, focus=None, radius=1):
    H = G
    if focus and focus in G:
        H = nx.ego_graph(G, focus, radius=radius)
    keep = [n for n, d in H.nodes(data=True)
            if (node_types is None or d["kind"] in node_types)
            and (categories is None or d["kind"] != "document" or d["category"] in categories)]
    H = H.subgraph(keep).copy()
    if relations is not None:
        H.remove_edges_from([(u, v) for u, v, d in H.edges(data=True) if d["relation"] not in relations])
    return H


OPTIONS = """
{
  "physics": {
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {"gravitationalConstant": -60, "centralGravity": 0.012,
                         "springLength": 90, "springConstant": 0.06, "avoidOverlap": 0.4},
    "stabilization": {"enabled": true, "iterations": 600, "fit": true},
    "minVelocity": 1
  },
  "edges": {"smooth": {"type": "continuous"}, "color": {"opacity": 0.55}},
  "nodes": {"font": {"size": 13, "strokeWidth": 3, "strokeColor": "#ffffff"}},
  "interaction": {"hover": true, "navigationButtons": false, "tooltipDelay": 120}
}
"""


def to_html(H, height="640px", highlight=(), focus=None, short_doc_labels=False):
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#222222",
                  cdn_resources="in_line", notebook=False)
    net.set_options(OPTIONS)
    highlight = set(highlight)

    for n, d in H.nodes(data=True):
        style = dict(NODE_STYLE[d["kind"]])
        if d["kind"] == "document":
            style["color"] = DOC_COLORS.get(d["category"], style["color"])
            tooltip = f"Doc {d['doc_id']} · {d['category']}\n{d['label']}\n\n{d['text']}"
        elif d["kind"] == "concept":
            tooltip = f"Concept · domain: {d['domain']}"
        else:
            tooltip = d["kind"].title()
        if n in highlight or n == focus:
            style["borderWidth"] = 4
            style["color"] = {"background": style["color"], "border": "#111111"}
            style["size"] = style["size"] + 6
        if d["kind"] != "document":
            label = d["label"]
        else:  # in big views only the id is shown; the title appears on hover
            label = f"{d['doc_id']}" if short_doc_labels else f"{d['doc_id']}. {d['label']}"
        net.add_node(n, label=label, title=tooltip, **style)

    for u, v, d in H.edges(data=True):
        style = EDGE_STYLE[d["relation"]]
        title = d["relation"] + (f" ({d['weight']})" if "weight" in d else "")
        net.add_edge(u, v, title=title, color=style["color"], dashes=style["dashes"], width=style["width"])

    # zoom to fit once the layout settles, so the whole graph is visible
    fit = ('<script>'
           'network.once("stabilizationIterationsDone", function () { network.fit({animation: false}); });'
           'setTimeout(function () { network.fit({animation: false}); }, 1500);'
           'setTimeout(function () { network.fit({animation: false}); }, 4000);'
           # the graph may be drawn inside a hidden tab (width 0): refit when it becomes visible
           'new ResizeObserver(function () { network.redraw(); network.fit({animation: false}); })'
           '.observe(document.getElementById("mynetwork"));'
           '</script>')
    return net.generate_html().replace("</body>", fit + "</body>")
