
# ArchRisk AI

LLM-Powered Software Architecture Integration Risk Assessment Using Graph Neural Networks

## Overview

ArchRisk AI analyzes software architecture descriptions to identify components whose failures could have the greatest impact on the system.

The system combines:
- Google Gemini for architecture extraction
- NetworkX for dependency graph construction
- Graph-based failure impact simulation
- PyTorch Graph Convolutional Network (GCN) for risk prediction
- Gemini for evidence-based recommendations
- Streamlit for interactive visualization

## Pipeline

Architecture Description
        |
        v
Gemini Component & Dependency Extraction
        |
        v
Directed Dependency Graph
        |
        v
Failure Impact Simulation
        |
        v
GNN Training & Evaluation
        |
        v
Risk Ranking & Gemini Recommendations

## Installation

Requires Python 3.10+ and a Google Gemini API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Set your Gemini API key in PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
$env:GEMINI_MODEL="gemini-3.6-flash"
```

Never commit API keys to GitHub.

## Run the Pipeline

```powershell
python archrisk.py
```

Quick test:

```powershell
python archrisk.py --variants 20 --epochs 20
```

Custom architecture:

```powershell
python archrisk.py --text-file data/architecture.txt
```

## Run the Dashboard

```powershell
streamlit run app.py
```

## Outputs

Results are saved in `results/`:

- `extracted_graph.json`: Extracted architecture graph
- `risk_scores.csv`: Component risk rankings
- `evaluation.json`: GNN evaluation metrics
- `recommendations.md`: Gemini-generated recommendations
- `risk_graph.png`: Dependency graph visualization
- `gnn_model.pt`: Trained GNN weights

## Risk Definition

A component's simulated risk is the fraction of other components that transitively depend on it.

A dependency A → B means A requires B.

The current simulation assumes that every dependency is mandatory and that failures propagate to all transitive consumers.

Risk scores represent simulated failure impact, not actual failure probability.

## Evaluation Limitations

The GNN is trained and evaluated on synthetic variations of a single architecture.

The evaluation measures performance on held-out synthetic graph variants, not generalization to completely independent software architectures.

This project is an academic prototype and should not be treated as a production reliability assessment tool.
