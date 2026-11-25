**Method.** Task-based walkthrough of the deployed interface (each task run end to end with an automated browser, actions counted) plus a heuristic review of three criteria: ease of interaction, interpretation of outputs and navigation between components. Server response times are the median of 20 runs.

| Task | Actions needed | Server time |
|---|---|---|
| Find documents about a problem (Recommend) | 2: type the query, click *Recommend* | ~4 ms |
| Get an answer from the chatbot | 1–2: click an example, or type + Enter | ~10 ms |
| Classify a new document | 2–3: pick an example or write, click *Classify* | ~1 ms |
| Explore a concept in the graph | 2: choose *Focus on a node*, pick the node | ~70 ms (+ 1–2 s layout in the browser) |
| Check the quality metrics | 1: open the *Evaluation* tab | ~3 s the first time, then cached |

**Ease of interaction.** Every tab works with defaults and has examples (sample questions, sample documents, VPN pre-selected in the graph), so a first-time user gets a result without typing. No task needs more than 3 actions.

**Interpretation of outputs.** Each result explains itself: recommendations show the score split into semantic, lexical and history parts plus the shared concepts and key terms; chatbot answers name the documents they come from and list every retrieved source with its score; the classifier shows its confidence and the words behind the decision; the graph has a legend and hover tooltips. Weak spot: raw scores (0.3–0.7) mean little to non-technical users.

**Navigation between components.** Tabs follow the natural order (overview → classify → recommend → ask → explore → evaluate). The sidebar history is visible from every tab and links them: a search in *Recommend* re-ranks the chatbot's retrieval and is highlighted in the graph. Weak spot: results can't be clicked to jump to another tab (e.g. open a chatbot source in the graph).

**Issues found during the walkthrough and fixed**
1. With history on, the chatbot answered an off-topic question (weather) with VPN documents → the out-of-scope check now always runs.
2. The whole graph opened zoomed in and cluttered → it now fits to the frame and big views show only document ids (title on hover).
3. The classifier listed words like *the* and *that* as evidence → articles and prepositions are hidden.
4. A policy question ("what should I do if I lose my phone?") was answered only with a past ticket → answers now combine a policy and a past ticket when both are retrieved.
5. After two VPN searches, asking for documents similar to *Phishing Email Report* returned VPN documents → the history boost is now scaled by each document's own similarity, so it re-ranks relevant results but never pushes unrelated ones up.

**Next step.** Run the 10-question SUS survey (docs/sus_questionnaire.md) with 5 employees and add the score here.
