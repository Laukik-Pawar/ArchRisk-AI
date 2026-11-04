


# ArchRisk AI — First Evidence

A runnable end-to-end **e-commerce microservices integration-risk** prototype.

## What it does

1. Architecture extraction: real OpenAI LLM (optional) or a verified reference JSON in demo mode.
2. Dynamic directed graph: NetworkX, edges mean `consumer -> required provider`.
3. Heuristic scores: normalized in/out degree and betweenness.
4. Failure simulation: failing a provider disables every transitive consumer (simplified mandatory-dependency assumption).
5. Graph Neural Network: two-layer message-passing GCN in **PyTorch**, trained to predict simulation risk from node type, declared criticality, description length, and graph structure. No degree/centrality values are supplied as explicit node features.
6. Evaluation: held-out synthetic variations of the *same* reference architecture.
7. Recommendations: grounded local templates or real OpenAI output in `llm` mode.
8. Visualization: PNG and optional Streamlit dashboard.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python archrisk.py --mode demo --variants 70 --epochs 75
streamlit run app.py
```

The CLI generates files in `results/`: `extracted_graph.json`, `risk_scores.csv`, `evaluation.json`, `recommendations.md`, `risk_graph.png`, and `gnn_model.pt`.

## Real LLM extraction and recommendations

Set your OpenAI API key in your local environment (never commit it):

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
python archrisk.py --mode llm --variants 70 --epochs 75
```

The default model is `gpt-4o-mini` (override with `ARCHRISK_MODEL`). API access is billed separately. The LLM output is checked for unique IDs and valid edge references, but extraction correctness still needs manual review.

**Important:** `demo` mode does not use an LLM. It reads the verified `data/architecture.json` regardless of edits to `data/architecture.txt` or the dashboard text area. This provides reproducibility and offline functionality. Use `llm` mode to process a changed text description.

## Architecture and assumptions

The single reference system has 15 components and 18 dependency edges. The direction `A -> B` means A depends on B. A failed node impacts its `nx.ancestors` (transitive consumers). We assume all declared dependencies are mandatory, and no redundancy or graceful degradation is modeled. This is **potential structural impact**, not measured outages, business loss, or real-world failure probability.

Risk targets are generated from simulated graph failures. The model learns a proxy label, not independently verified incident outcomes. The heuristic score is `.3*normalized_indegree + .3*normalized_outdegree + .4*betweenness` and is a baseline only.

## GNN evaluation limitations

Synthetic training graphs retain the same 15 component identities and types and modify dependency edges. The 80/20 split is by graph variant, not by node. Therefore held-out scores indicate performance on variations of one architecture, **not** generalization to entirely new architectures. The model has only two message-passing layers, so it may not capture long dependency chains accurately. More independent architecture graphs and richer failure assumptions are needed for the final submission.

## Suggested first-evidence demo

1. Show `data/architecture.txt` and the corresponding verified JSON.
2. Run `python archrisk.py --mode demo --variants 70 --epochs 75`.
3. Open `results/risk_graph.png`, `risk_scores.csv`, `evaluation.json`.
4. Start the dashboard and select `Message Queue` under Failure simulation.
5. Show grounded recommendations and explain why a single reference graph limits generalization.
6. If an API key is available, switch to Live LLM mode and compare extracted edges with the verified reference.

## Final submission roadmap

Add PDF/DOCX/YAML ingestion, robust LLM extraction validation, independent architecture graphs for training/testing, richer propagation rules, more comprehensive evaluations, and grounded architecture-specific recommendations.
