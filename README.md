# ArchRisk AI

**LLM-Powered Software Architecture Integration Risk Assessment with Graph Neural Networks**  
**Stage:** First Evidences academic prototype · **Example domain:** E-commerce microservices  
**Repository:** https://github.com/Laukik-Pawar/ArchRisk-AI

## What the project does

ArchRisk AI accepts a written description of a software architecture and identifies components whose failure could disrupt other components. It uses **Google Gemini** to extract components and dependencies, **NetworkX** to construct a directed dependency graph, a deterministic **failure-impact simulation** to produce risk labels, a lightweight **PyTorch GNN** to learn those labels, and **Gemini** to draft evidence-grounded engineering recommendations. A **Streamlit dashboard** visualizes the results.

> **Interpretation:** The risk score measures simulated *failure impact*, **not** the probability that a component will fail. The first-evidence experiment uses one e-commerce architecture and synthetic variants; it is not a production reliability assessment.

## Quick-start for a teammate (Windows PowerShell)

### 1. Prerequisites

- **Python 3.11 or 3.12 recommended** (the original developer also ran Python 3.13). Ensure Python and `pip` are available.
- **Git** installed: https://git-scm.com/downloads
- Internet access for dependency installation and Gemini API calls.
- A Google account and a Gemini API key.

Verify:

```powershell
python --version
git --version
```

If `python` is not recognized, install Python from https://www.python.org/downloads/ and reopen PowerShell. On some Windows systems, `py` works instead of `python`.

### 2. Clone the project

```powershell
git clone https://github.com/Laukik-Pawar/ArchRisk-AI.git
cd ArchRisk-AI
```

If your friend received the project as a ZIP instead, extract it and open PowerShell inside the extracted folder.

### 3. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in the terminal prompt. If PowerShell blocks activation, run this **for the current PowerShell process only** and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install project dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include `google-genai`, `python-dotenv`, `networkx`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `matplotlib`, `streamlit`, and `plotly`. Installing PyTorch may take a while.

**Do not type `python-dotenv>=1.0` directly in PowerShell.** That line belongs inside `requirements.txt` and is installed through `pip install -r requirements.txt`.

### 5. Obtain your own Gemini API key

1. Visit **Google AI Studio**: https://aistudio.google.com/apikey
2. Sign in with your Google account and accept any applicable terms.
3. Choose **Create API key** and select or create a Google Cloud project if prompted.
4. Copy the generated key. Keep it private. Google may require billing setup or impose quota limits depending on the account, region, and model.
5. **Do not paste the key into GitHub, screenshots, chats, or your presentation.**

The project was tested with `gemini-3.6-flash`. Available model names and quotas can vary by account. If it is not available, list models using the Google Gen AI SDK and select an available text-generation model.

### 6. Create the local `.env` configuration

Create a file named **`.env`** in the project root (next to `archrisk.py`), with the following content:

```dotenv
GEMINI_API_KEY=PASTE_YOUR_OWN_KEY_HERE
GEMINI_MODEL=gemini-3.6-flash
```

Replace the placeholder with your own key. **Do not add quotation marks unless needed by your environment.** The repository's `.gitignore` excludes `.env` and `.env.*`; keep it that way.

`archrisk.py` calls `load_dotenv()` so the key and model are loaded automatically by both the CLI and the Streamlit app. You do **not** need to re-enter `$env:GEMINI_API_KEY` each time you open a terminal. An already-set shell environment variable normally takes precedence over the `.env` value.

To verify the configuration **without exposing your key or making an API request**:

```powershell
python -c "import archrisk, os; print('Key loaded:', bool(os.getenv('GEMINI_API_KEY'))); print('Model:', os.getenv('GEMINI_MODEL'))"
```

Expected: `Key loaded: True` and `Model: gemini-3.6-flash`.

### 7. Run the interactive dashboard (recommended for the presentation)

```powershell
python -m streamlit run app.py
```

Open the local address printed in the terminal, typically **http://localhost:8501**. The dashboard loads the reference e-commerce description by default. Click **Analyze architecture** and wait for Gemini extraction, GNN training, and Gemini recommendations to complete. The dashboard then shows:

- Component and dependency counts
- Simulated-risk and GNN-predicted-risk rankings
- Interactive dependency graph
- What-if failure impact for a selected component
- Engineering recommendations and evaluation metrics
- CSV and JSON downloads

**The dashboard retrains on 70 synthetic graph variants for 75 epochs.** Its metrics can differ from the full CLI experiment below. Each analysis requires Gemini API access and can be delayed by provider capacity.

### 8. Run the command-line pipeline

For a short smoke test:

```powershell
python archrisk.py --variants 20 --epochs 20
```

For the recorded full experiment:

```powershell
python archrisk.py --variants 120 --epochs 160
```

To analyze a different **text** architecture description:

```powershell
python archrisk.py --text-file data/architecture.txt
```

The default input is `data/architecture.txt`. The current CLI reads a plain-text file; it does **not** directly parse PDF files, diagrams, or GitHub repository URLs. The first-evidence evaluation is scoped to the reference architecture.

### 9. Inspect generated outputs

After a successful CLI run, look in `results/`:

| File | Purpose |
|---|---|
| `extracted_graph.json` | Gemini-extracted components and dependencies |
| `risk_scores.csv` | Simulated, heuristic, and GNN-predicted scores |
| `evaluation.json` | GNN and baseline evaluation metrics |
| `recommendations.md` | Gemini's evidence-based suggestions |
| `risk_graph.png` | Static architecture graph |
| `gnn_model.pt` | Saved trained GNN parameters |

**Running the quick CLI test overwrites these output files with quick-test results.** To reproduce the reported full evaluation, run the 120-variant/160-epoch command last.

### 10. Run automated tests (no Gemini key or API call required)

```powershell
python -m unittest test_archrisk.py -v
```

Five tests check architecture validation, duplicate IDs, graph construction, unknown dependencies, and transitive failure propagation. The tests are expected to finish quickly and report `OK`.

## Pipeline and scoring

```text
Architecture description (plain text)
       |
       v
Google Gemini -> validated JSON components + dependencies
       |
       v
NetworkX directed graph: consumer -> required provider
       |
       +--> Transitive failure simulation -> target risk labels
       |
       +--> Synthetic edge variants -> train/test split
                                      |
                                      v
                              Two-layer PyTorch GCN
                                      |
                                      v
                           Risk predictions + metrics
                                      |
                                      v
                      Gemini recommendations + dashboard
```

For a graph with `N` components, the simulated failure impact of component `v` is:

`impact(v) = number of other components transitively dependent on v / (N - 1)`.

The model assumes each documented dependency is mandatory and failure propagates to all upstream consumers. Real systems with redundancy, failover, partial degradation, or optional dependencies may behave differently.

## Recorded evaluation (full CLI run)

The following values are **recorded results from one run**, not independently reproduced benchmarks:

| Metric | Recorded value |
|---|---:|
| Extracted components / dependencies | 15 / 18 |
| Training / held-out synthetic graphs | 96 / 24 |
| Held-out node observations | 360 |
| GNN MAE / RMSE | 0.04246 / 0.06191 |
| GNN Spearman | 0.89226 |
| In-degree baseline MAE / Spearman | 0.06310 / 0.83034 |
| Original heuristic Spearman | 0.04057 |

The 360 observations come from variants of **one** architecture, not 360 independent real-world systems. The GNN outperformed the two reported baselines in this run; the simulation labels are directly computable from the graph, so a GNN is being evaluated as an experimental approximation rather than a necessary substitute for the exact simulation.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `GEMINI_API_KEY is required` | Confirm `.env` is in the project root, contains the correct variable name, and `python-dotenv` is installed. |
| `503 UNAVAILABLE` / high demand | The Gemini server is overloaded. The code retries temporary server errors; wait and retry later or choose another model listed for your account. |
| `404` / model not found | Check `GEMINI_MODEL` against models your API key can access. |
| `ModuleNotFoundError` | Activate `.venv`, then run `python -m pip install -r requirements.txt`. |
| PowerShell won't activate `.venv` | Use the process-scoped execution policy command shown above. |
| `invalid int value: '20+'` | Use `--epochs 20` (an integer without `+`). |
| Dashboard takes time | Gemini calls and GNN training are synchronous; wait for completion and watch the terminal for retry messages. |
| Different metrics from report | Confirm whether you used the dashboard's 70/75 settings, the CLI quick 20/20 settings, or the full CLI 120/160 settings. |
| `.env` accidentally committed | Remove it from Git tracking, revoke the exposed API key, and create a replacement. |

## Current scope and limitations

- Gemini is mandatory; there is **no OpenAI or offline/demo mode** in the current pipeline.
- The first-evidence example is an e-commerce microservices architecture described in text.
- A two-layer GCN has limited receptive field for long dependency chains.
- Train/test graphs are synthetic variations of the same components; transfer to unrelated architectures has **not yet been tested**.
- The system predicts a simulated impact proxy, **not** empirical outage likelihood, incident cost, or guaranteed operational risk.
- Google Gemini's extraction can vary across calls; inspect extracted JSON before relying on conclusions.

See [`METHODOLOGY.md`](METHODOLOGY.md) for the full design history, experiments, results, research context, and next steps.

## Security and collaboration

Never share `.env`, API keys, or credentials. Every collaborator should create their **own** Gemini API key. Commit only code, documentation, non-secret architecture examples, and safe test artifacts. If a key has ever been exposed, revoke and rotate it.
