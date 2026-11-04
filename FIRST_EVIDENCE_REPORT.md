# ArchRisk AI — First Evidence Report

## Project objective

Build an AI-assisted tool to identify components in an e-commerce software architecture whose failure could affect many dependent services. The first evidence uses **one reference microservices architecture**. The final project will expand to new architecture files.

## Method

- **Architecture:** 15 components and 18 directed dependencies. `consumer -> provider` defines the direction of each edge.
- **Extraction:** The local demo loads a verified JSON graph; an optional OpenAI-powered mode extracts components and dependencies from a text description.
- **Graph:** NetworkX constructs a directed graph and calculates degree and betweenness metrics.
- **Failure simulation:** Each component is removed conceptually; every transitive consumer is considered affected. This assumes all dependencies are required and no fallback exists.
- **Risk labels:** A component's target is the fraction of *other* components affected by its simulated failure.
- **GNN:** A two-layer PyTorch message-passing GCN predicts simulation risk using component attributes and graph structure. Centrality/degree are not supplied as explicit node features.
- **Evaluation:** Synthetic edge variations of the same architecture are split into training and test graphs. We compare predicted scores with simulation labels and heuristic ranking.
- **Feedback:** Local templates provide grounded engineering suggestions; optional OpenAI mode can draft suggestions from ranked evidence.

## What the prototype demonstrates

The project can build a dependency graph, calculate heuristic risk, estimate transitive failure impact, train a basic GNN, display predicted rankings, and produce engineering suggestions. Outputs are saved in the `results` folder.

## Important limitations

This is an experimental architecture-analysis prototype. Its results are **not real-world failure probabilities**. The simple propagation model can overestimate impact. The synthetic train/test split does not validate generalization to independent architectures. Live LLM extraction needs an API key and manual validation. The first-evidence demo does not yet process arbitrary uploaded PDFs or architecture diagrams.

## Final phase

Support multiple architecture files, improve extraction validation, train on independent architecture graphs, test on unseen systems, and expand the Streamlit interface.
