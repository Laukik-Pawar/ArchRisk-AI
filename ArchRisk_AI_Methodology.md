# ArchRisk AI — Detailed Methodology, Design Decisions, and Experimental Record

**Project:** LLM-Powered Software Architecture Integration Risk Assessment Using Graph Neural Networks  
**Phase:** Project 2 — First Evidences  
**Case study:** E-commerce microservices architecture  
**Repository:** https://github.com/Laukik-Pawar/ArchRisk-AI  
**Record prepared:** November 2026

> **Evidence policy:** This report separates implemented features, observed tests, conceptual research, and proposed future work. It does not claim an external literature review, production outage dataset, or cross-architecture experiment that has not been performed.

## 1. Problem statement and motivation

Modern software systems consist of interdependent services, databases, queues, caches, gateways, and external integrations. The failure of one shared provider can affect multiple consumers, including consumers that depend on it indirectly. A flat list of components does not capture these transitive effects.

The project asks: **Can an LLM extract a dependency graph from an architecture description, and can a graph neural network approximate the relative failure impact of its components?** A secondary question is whether the resulting graph evidence can support useful engineering recommendations.

The prototype focuses on **integration risk as simulated downstream disruption**. It does not estimate component failure probability or validate predictions against real outage incidents.

### Research questions

- **RQ1 — Extraction:** Can Gemini turn a natural-language architecture description into a structured, internally consistent dependency graph?
- **RQ2 — Impact:** Which components have the greatest simulated transitive impact under an all-dependencies-required assumption?
- **RQ3 — Learning:** How accurately does a lightweight GNN approximate graph-derived impact labels on held-out synthetic variations of the same architecture?
- **RQ4 — Baselines:** Does the GNN improve on simple in-degree and hand-weighted structural heuristics?
- **RQ5 — Explainability:** Can Gemini generate recommendations tied to measured graph evidence rather than unsupported reliability claims?

RQ1 has been demonstrated on one reference input, not quantitatively benchmarked over a labeled extraction dataset. RQ3 and RQ4 have one recorded evaluation. RQ5 has a working implementation, but recommendation quality has not undergone a blinded or expert-scored study.

## 2. Scope selection and architecture choice

### Why e-commerce?

The first-evidence scope deliberately uses **one comprehensible microservices architecture**, not arbitrary repositories or PDF diagrams. E-commerce provides familiar component roles and both direct and transitive dependencies, while remaining small enough to inspect manually and present in a classroom.

The reference is `data/architecture.txt`. It describes an API gateway, authentication, catalog, cart, orders, payments, inventory, notifications, data stores, cache, a message queue, and an external payment provider. The recorded extraction contains **15 components and 18 directed dependencies**.

This architecture is an illustrative, manually written case study, **not a claim of a real company's production topology**. No real outage records or component failure rates were used.

### Deliberate simplifications

1. Every declared dependency is mandatory.
2. A provider failure disables all transitive consumers.
3. There is no redundancy, graceful degradation, circuit-breaker behavior, recovery-time model, or partial service availability.
4. Every component is counted equally in the impact fraction.
5. The first evidence operates on natural-language text, not uploaded diagrams or source-code analysis.

These assumptions make the label deterministic and reproducible, but they can overstate operational risk.

## 3. Technology and strategy selection

| Concern | Chosen approach | Rationale and tradeoff |
|---|---|---|
| Architecture extraction | **Google Gemini** (`google-genai`) | Converts unstructured architecture prose into JSON without hardcoded component rules; depends on a live API and can make extraction errors. |
| API model | **`gemini-3.6-flash`** for successful recorded runs | Available in the developer's model list and completed a full run after `gemini-3.8-flash` experienced repeated 503 errors. This is an operational selection, **not** a measured head-to-head accuracy benchmark. |
| Configuration | `.env` + `python-dotenv` | Keeps the key and model out of source code and avoids repeated terminal entry. |
| Graph engine | NetworkX `DiGraph` | Straightforward directed topology, centrality, and transitive ancestor queries. |
| Label generation | Deterministic graph simulation | Provides a transparent target without requiring unavailable real failure labels. |
| Learned model | Two-layer PyTorch GCN-style message passing | Small enough to train and explain in first evidence; may miss longer-range structure. |
| Evaluation | Synthetic graph variants, held-out split, baselines | Enables an initial controlled experiment; does not establish transfer to unrelated systems. |
| Recommendations | Gemini prompted with computed evidence | Natural-language suggestions grounded in top-ranked components; not an independently validated engineering audit. |
| UI | Streamlit + Plotly | Interactive graph, risk ranking, failure selection, downloads, and quick classroom demonstration. |

### Gemini selection and reliability investigation

The implementation was intentionally simplified to a **single Gemini-based LLM path**, rather than parallel OpenAI and offline/demo modes. This reduced divergent behavior and kept the demonstration consistent with the intended architecture.

Observed engineering sequence:

1. The team implemented Gemini JSON extraction and recommendation generation.
2. Early integration encountered an SDK client-lifetime error (`Cannot send a request, as the client has been closed`). Keeping a named client instance through the call addressed the issue.
3. Model availability was account-dependent; a previously attempted model was unavailable to the API key. The developer listed accessible models using the SDK.
4. `gemini-3.8-flash` repeatedly returned **HTTP 503 / high demand**, even after retries.
5. A shared retry helper was added for temporary server errors **500, 502, 503, 504**, with exponential backoff plus jitter, up to five application-level attempts. The SDK may also perform internal retries.
6. `gemini-3.6-flash` was selected and a complete end-to-end run succeeded, despite transient 503 responses at the start.
7. `.env` loading was added to simplify local use and protect the key from accidental source-code commits.

**Important distinction:** This is evidence of *API compatibility and practical availability*, not evidence that `gemini-3.6-flash` is more accurate, faster, or cheaper than other models. No formal latency, cost, extraction F1, or cross-model comparison was recorded.

### Security strategy

The API key is read from `GEMINI_API_KEY`; the model from `GEMINI_MODEL`. The `.gitignore` excludes `.env`, `.env.*`, and virtual environments. Each collaborator should obtain their own key. Any key previously shared publicly should be revoked and replaced. The current repository code has two definitions of `gemini_client()`; Python uses the later definition. This duplication is a cleanup item, not a demonstrated runtime blocker.

## 4. End-to-end processing pipeline

```text
Natural-language architecture text
             |
             v
Gemini extraction (JSON response)
             |
             v
Schema and graph-reference validation
             |
             v
NetworkX directed dependency graph
             |
             +--> Exact transitive impact simulation --> labels
             |
             +--> Graph variants --> 80/20 train/test split
                                      |
                                      v
                             Two-layer GNN training
                                      |
                                      v
                      Predicted impact + baseline metrics
                                      |
                                      v
                        Evidence-limited Gemini advice
                                      |
                                      v
                       CLI outputs / Streamlit dashboard
```

### 4.1 LLM extraction and prompt constraints

`extract_architecture(text)` instructs Gemini to extract only explicitly described components and dependencies, returning JSON with:

- `system`: system name
- `edge_semantics`: relationship interpretation
- `components`: objects containing `id`, `name`, `type`, `description`, `criticality`
- `dependencies`: directed `source`, `target` pairs

Allowed types: `gateway`, `service`, `database`, `cache`, `queue`, `external`, `other`. The prompt requests snake_case unique IDs, criticality in `[0,1]`, default `0.5` when unspecified, and **no invented dependencies or redundancy**. The Gemini SDK requests `response_mime_type="application/json"` and `temperature=0`.

The response is parsed using `json.loads` and checked by `validate()`. Validation rejects fewer than two components, missing or duplicate IDs, invalid types, out-of-range criticality, unknown endpoints, and self-dependencies. It checks structural consistency; it **does not prove semantic faithfulness** to the source description. Manual inspection of extracted graph JSON is still necessary.

### 4.2 Dependency graph semantics

`build_graph()` constructs a `networkx.DiGraph` with one node per component. Each edge follows:

`consumer -> required provider`

For example, `Order Service -> Message Queue` means the Order Service requires the queue. If the queue fails, the disruption propagates **against the edge direction** toward all transitive consumers.

This orientation is essential: NetworkX's `ancestors(G, node)` returns all nodes with a directed path **to** the failed provider, exactly the affected-consumer set under this simplified model.

### 4.3 Risk labels and failure simulation

For component `v` in a graph of `N` nodes:

`simulation_risk(v) = |ancestors(G, v)| / (N - 1)`

The denominator excludes the failed component itself. For `N=15`, a component affecting five other components receives `5/14 = 0.357143`.

This is a deterministic **impact score**, not a failure probability. The algorithm effectively assumes every upstream component loses functionality if any required provider fails. There is no random failure sampling or empirical incident history in the present simulation.

The implementation also records `affected_components`, in-degree, out-degree, and normalized betweenness centrality. Its original heuristic is:

`heuristic = 0.3 * normalized_in_degree + 0.3 * normalized_out_degree + 0.4 * betweenness`

This hand-weighted formula is a comparison baseline; its weights were not statistically fitted to failure-impact labels.

### 4.4 Synthetic training data

`make_variants()` begins with the reference graph and generates 120 variants by:

1. Copying the same set of nodes and their attributes.
2. Independently removing each original edge with probability **0.18**.
3. Attempting between **0 and 4** additional randomly selected edges, subject to cycle-avoidance and duplicate-edge checks.
4. Recomputing exact transitive-impact labels for each variant.

The code uses NumPy's random generator with seed **42**. It splits the generated graphs in order: **80% training and 20% testing** (96/24 for 120 variants). This is a synthetic topology-perturbation experiment, not a dataset of independently designed systems. The same component identities, names, and attributes appear in both sets, creating a significant limit on claims of generalization.

### 4.5 GNN input features and architecture

For each node, `features()` constructs **9 numeric features**:

- Seven-way one-hot encoding of component type
- Criticality (`0` to `1`)
- Description-length feature, clipped at 100 characters and normalized

It does **not** directly feed degree, betweenness, or the simulation target into the GNN. Structural information is supplied through a normalized adjacency matrix. Because edges are consumer-to-provider, the adjacency is reversed for message aggregation so providers can receive information from their consumers. Self-loops are added and rows are normalized.

`SimpleGCN` applies two rounds of neighborhood aggregation and linear transformation:

`H1 = ReLU(W1 * (A X) + b1)`  
`H2 = ReLU(W2 * (A H1) + b2)`  
`predicted_impact = sigmoid(Wout * H2 + bout)`

The hidden width is **32**, and the output is one number per node in `[0,1]`. This is a lightweight GCN-style network, not a library-based graph transformer.

### 4.6 Optimization and evaluation

`train_and_evaluate()` uses:

- Adam optimizer, learning rate **0.015**, weight decay **0.0001**
- **160 epochs** for the full CLI experiment
- Mean squared error between predicted and simulated impact
- Seed **42** for NumPy and PyTorch
- Held-out synthetic graphs for evaluation

Reported metrics include:

- **MAE**: average absolute difference between predicted and simulated scores
- **RMSE**: root mean squared difference, penalizing larger errors
- **Spearman correlation**: agreement between predicted and simulated rankings
- **In-degree baseline**: normalized number of direct consumers, measured with MAE and Spearman
- **Original heuristic**: measured with Spearman

The in-degree baseline is particularly relevant because the number of direct consumers is a simple proxy for potential disruption. Spearman can be unstable or undefined when a prediction vector is constant; the code converts nonfinite correlations to `null` in JSON.

## 5. Observed tests and results

### 5.1 End-to-end smoke test

Command:

```powershell
python archrisk.py --variants 20 --epochs 20
```

Observed extraction: **15 components, 18 dependencies**. The run completed and saved outputs after transient Gemini 503 retries.

| Metric | Quick-test result |
|---|---:|
| Training / test graphs | 16 / 4 |
| Test node observations | 60 |
| GNN MAE | 0.10166 |
| GNN RMSE | 0.13792 |
| GNN Spearman | 0.36093 |
| Original heuristic Spearman | 0.20925 |

This run established pipeline functionality, not strong predictive performance. Its outputs were superseded by the full experiment.

### 5.2 Full synthetic-variant evaluation

Command:

```powershell
python archrisk.py --variants 120 --epochs 160
```

Recorded from the successful run and committed `results/evaluation.json`:

| Metric | Recorded value |
|---|---:|
| Components / dependencies | 15 / 18 |
| Training graphs | 96 |
| Test graphs | 24 |
| Test node observations | 360 |
| GNN MAE | **0.0424633** |
| GNN RMSE | **0.0619075** |
| GNN Spearman | **0.8922617** |
| In-degree baseline MAE | **0.0630952** |
| In-degree baseline Spearman | **0.8303432** |
| Original heuristic Spearman | **0.0405684** |

Relative to the in-degree baseline, the GNN's MAE was approximately **32.7% lower** in this run. The GNN also ranked components more consistently with simulation labels than either reported baseline.

**Interpretation limits:** This is a single seeded synthetic experiment. There are 360 *node observations*, but only 24 held-out synthetic graphs, all variants of the same 15-node case study. No confidence intervals, repeated-seed variance, statistical significance test, or independent-architecture evaluation has been completed. Furthermore, the exact simulation target is computable directly from the graph; a GNN is not required to obtain these labels. The learning experiment demonstrates approximation ability under the chosen synthetic distribution.

### 5.3 Example risk ranking

From the full run, the highest-impact components were:

| Component | Simulated impact | GNN predicted impact |
|---|---:|---:|
| Message Queue | 0.357143 | 0.278303 |
| Inventory Database | 0.285714 | 0.207024 |
| Cache Service | 0.214286 | 0.207924 |
| Inventory Service | 0.214286 | 0.159189 |
| Order Database | 0.214286 | 0.239892 |
| Payment Provider | 0.214286 | 0.180429 |

The Message Queue affects **5 of 14** other components under the simulation assumptions. The scores should not be interpreted as measured probabilities of failure.

### 5.4 Automated unit tests

Command:

```powershell
python -m unittest test_archrisk.py -v
```

Five tests passed (`OK`, approximately 0.001 s in the recorded local run):

1. Valid architecture accepted.
2. Duplicate component IDs rejected.
3. Directed dependency graph constructed correctly.
4. Transitive failure propagation gives expected impact on a three-node chain.
5. Dependency to a nonexistent component rejected.

These tests do **not** call Gemini, validate extraction accuracy against a gold standard, or test model performance across unrelated architectures.

### 5.5 What was investigated but not yet evaluated

- **Cross-architecture generalization:** A banking-architecture holdout was proposed as Fix 8, but **not executed** before First Evidences.
- **Gemini model benchmarking:** No controlled accuracy/latency/cost comparison across models.
- **Real-world outage labels:** No incident dataset or historical outage ground truth.
- **LLM extraction scoring:** No manually labeled benchmark corpus or precision/recall/F1.
- **Expert evaluation of recommendations:** No rubric-based assessment yet.
- **Ablations:** No formal comparison of GNN depth, node features, or graph direction variants.

## 6. Dashboard and deliverables

The Streamlit application (`app.py`) loads the reference text, invokes Gemini extraction, builds the graph, computes simulation risk, trains a GNN, generates Gemini recommendations, and displays results interactively.

**Dashboard training settings differ from the full CLI evaluation:** the dashboard uses **70 synthetic variants and 75 epochs** for a shorter demonstration. Its metrics should not be presented as the 120/160 full-run results unless the dashboard code is changed and rerun accordingly.

The CLI saves `results/extracted_graph.json`, `risk_scores.csv`, `evaluation.json`, `recommendations.md`, `risk_graph.png`, and `gnn_model.pt`. The dashboard supports CSV and JSON downloads of the current analysis. `README.md` documents fresh installation and API key configuration.

## 7. Research framing and relevant methodological concepts

The design draws on established *categories* of methods, rather than claiming a completed systematic literature review:

- **Software dependency analysis:** Model software services and infrastructure as a directed graph to expose shared providers and dependency chains.
- **Failure propagation / reachability:** Use graph reachability to approximate which consumers may be disrupted by a provider failure.
- **Graph neural networks:** Learn node-level predictions by aggregating neighborhood information over a graph.
- **Synthetic data augmentation:** Perturb graph edges to create supervised examples when labeled incident data is unavailable.
- **Baseline comparison:** Compare learned predictions against transparent structural statistics rather than reporting standalone model metrics.
- **LLM information extraction:** Use constrained prompts and JSON validation to transform prose into structured records.
- **Grounded generation:** Provide computed graph evidence to the recommendation model and forbid unsupported claims about reliability.

These concepts should be supported with peer-reviewed references in the final report. No particular paper is asserted here as having been read or experimentally replicated. Suggested foundational literature to verify and cite in the final submission includes Kipf & Welling's graph convolutional networks and established work on graph-based software dependency analysis and reliability engineering.

## 8. Threats to validity

**Construct validity:** The label is *transitive reachability under mandatory dependencies*, not operational risk in its broad sense. It omits outage frequency, severity weighting, redundancy, duration, recovery, and user impact.

**Internal validity:** The synthetic perturbation generator determines the data distribution. The GNN may exploit repeated component attributes and similar graph structures rather than learning generally transferable patterns. The hand-weighted heuristic is not optimized for the target.

**External validity:** The 80/20 split separates synthetic variants, **not independent architectures**. Results do not establish performance on banking, healthcare, logistics, or real production systems.

**Measurement validity:** Gemini extraction is not scored against human-labeled truth. A syntactically valid graph can still be semantically incorrect. Recommendation quality is not independently rated.

**Reproducibility:** NumPy and PyTorch seeds are set for graph generation/training, but Gemini responses and API availability can vary. Different extracted graphs can change every downstream metric. Package versions use minimum bounds rather than a fully locked environment.

**Architectural limitations:** The two-layer GNN aggregates information only over a limited number of hops, while the target depends on all transitive consumers. Longer chains may be underrepresented.

## 9. Engineering lessons and debugging record

| Observation | Diagnosis / action |
|---|---|
| Gemini client-closed error | Keep a named SDK client alive while making the request. |
| Gemini model not found | Query the accessible model list instead of assuming a model ID works for every account. |
| Repeated `503 UNAVAILABLE` | Introduce bounded retry with backoff/jitter and use an available Flash model. |
| Extraction response-handling bug | Remove an accidental trailing comma that created a tuple; parse `response.text` and return validated JSON. |
| Wrong prompt variable | Ensure extraction passes the architecture instruction and text, not an undefined recommendation prompt. |
| `--epochs 20+` parse error | CLI argument must be an integer, e.g., `--epochs 20`. |
| Key re-entry on each launch | Use `python-dotenv` and a Git-ignored `.env` file. |
| Initial small-run GNN metrics were weak | Increase to the intended 120 variants and 160 epochs; record separate results rather than treating smoke-test metrics as final. |
| Need stronger evaluation | Add an in-degree baseline, compare MAE and ranking correlation, and explicitly disclose single-architecture limitations. |
| Need reproducible graph logic | Add five no-API unit tests and push them to GitHub. |

The current code contains two `gemini_client()` definitions; the second overrides the first. Consolidation is recommended after the presentation. The older `FIRST_EVIDENCE_REPORT.md` still mentions OpenAI/demo modes and should be replaced or corrected before formal submission; the current code is Gemini-only.

## 10. First Evidences presentation plan

Suggested presentation flow:

1. **Problem:** A failure in a shared provider can affect multiple services.
2. **Input:** Show the e-commerce architecture description.
3. **LLM extraction:** Show the 15-component, 18-dependency graph and explain edge direction.
4. **Simulation:** Select Message Queue and explain its five affected consumers.
5. **GNN:** Explain node features, two rounds of message passing, and synthetic training variants.
6. **Evidence:** Present the 120/160 evaluation table, including the in-degree baseline.
7. **Limitations:** Explain mandatory-dependency assumptions, synthetic-only evaluation, and untested cross-architecture transfer.
8. **Next phase:** Evaluate independent architectures, validate extraction against annotations, and improve robustness.

**Live-demo reliability:** Gemini API availability is external. Run a smoke test before presenting, preserve safe result artifacts and screenshots for backup, and do not claim that stored outputs represent a live Gemini run. The current dashboard has no offline mode.

## 11. Future work — explicitly not completed

**Short term (after First Evidences):**

- Execute the proposed **cross-architecture banking test** without retraining; compare MAE and Spearman to the in-domain results.
- Expand evaluation to multiple independently authored architectures and prevent shared-identity leakage between training and test systems.
- Build a manually annotated extraction benchmark and measure component/edge precision, recall, and F1.
- Add tests for invalid JSON, missing attributes, malformed dependency endpoints, and Gemini errors using mocks.
- Remove duplicate `gemini_client()` and obsolete OpenAI/demo references.

**Later / final project:**

- Explore longer-range GNN message passing, attention, or alternative graph models and perform ablation studies.
- Compare deterministic simulation, in-degree, other graph baselines, and GNN across multiple seeds.
- Introduce weighted criticality, optional dependencies, redundancy, failover, and partial degradation.
- Study incident-informed or failure-injection labels if trustworthy datasets become available.
- Evaluate recommendation correctness with a human-reviewed rubric.
- Support additional text architectures and, only after separate parser work, richer file formats or repository-derived architecture evidence.

## 12. Reproduction commands

```powershell
# From the project root, after creating .venv and .env:
python -m pip install -r requirements.txt

# Graph unit tests (no Gemini request)
python -m unittest test_archrisk.py -v

# Quick end-to-end test
python archrisk.py --variants 20 --epochs 20

# Full recorded experiment settings
python archrisk.py --variants 120 --epochs 160

# Interactive dashboard
python -m streamlit run app.py
```

**Conclusion:** The First Evidences prototype demonstrates a working Gemini-to-graph-to-GNN workflow and a promising synthetic evaluation against transparent baselines. Its main scientific limitation is that both training and testing are based on variations of one hand-authored architecture. Independent-architecture validation remains the next substantive research step.
