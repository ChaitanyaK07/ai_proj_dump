# AI-usage log

The course requires disclosure of AI assistance. Add one row per use: what was asked, what was produced,
and how a team member checked it. Unchecked AI output must not be merged.

| Date | Member | Tool | What it was used for | How it was verified |
|---|---|---|---|---|
| 2026-10-18 | Chaitanya Srikar Kastury | Claude Code (Claude Opus 5.5) | Scaffolded the full implementation from the proposal: package layout, graph/feature/cost modules, Dijkstra / A* / weighted A* / Focal Search / backward Dijkstra, label pipeline, node2vec, heuristic network and training, evaluation harness, FastAPI service, web UI, user-study tooling, tests and docs. | Test suite (`python -m pytest`): every search compared against exact optima (networkx reference and backward Dijkstra), Focal bound checked with an adversarial guide, `h_adm` admissibility/consistency checked on every edge; full pipeline run on the real graph. **Each pair must still review and be able to explain its own components before the viva.** |
