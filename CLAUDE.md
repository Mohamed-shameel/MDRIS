# CLAUDE.md — MDRIS guardrails

Read `docs/MDRIS_CONTEXT.md` in full before any work. It is the specification; this file holds the rules.

## Project in one line
Flood-only decision support: fine-tuned YOLOv8n (vision) + fine-tuned DistilBERT (3-level urgency) + a documented
confidence/reliability-weighted fusion rule. FastAPI backend, React frontend. Applied Deep Learning course project.

## Hard rules (never break)
1. **No fake outputs.** Never add mock, fallback, placeholder or heuristic results to make something "work".
   If a model or file is missing, fail loudly: log at ERROR and return HTTP 503/422 with the reason.
2. **No silent CPU training.** Training scripts must assert `torch.cuda.is_available()` and exit if it is false.
3. **Do not improvise decisions.** Class names, class weights, the urgency mapping, fusion constants, splits,
   hyperparameters and API schemas are fixed in docs/MDRIS_CONTEXT.md. If reality does not match (unknown class
   names, too few rows, missing files), STOP and report. Do not guess.
4. **Scope lock.** Flood only. No database, auth, Docker, WebSockets or new pages beyond Section 11.
5. **Weather never enters fusion.** It is display-only.
6. **Every API response includes `data_origin`** (`model` | `simulated` | `static_demo` | `live` | `external_llm`).
   `external_llm` is only the Groq second opinion (D-36): always labelled with the LLM's name, never as YOLO, never fused.
7. **Git:** work on branch `rebuild` only. Never commit to `main`, never force-push, never rewrite history.
   One commit per module, message prefix `M<n>:`. Never commit `datasets/`, `.env` or large NLP weights.
8. **Never delete or modify `datasets/raw/`.** Write derived data to `datasets/processed/`.
9. **One GPU (RTX 2050, 4 GB).** Never run two trainings at once. YOLO: batch 8 (4 on OOM), workers 2 (0 if it hangs).
   Long training runs go in the background with a log file that you poll.
10. **Windows-safe code:** `pathlib` everywhere, `if __name__ == "__main__":` in every training script, no bash-only syntax
    in scripts the user runs.
11. **Stop at human checkpoints:** after M1 (data spot-check), after M2 and M3 (metrics review). Report and wait.

## Module order
M0 scope strip → M1 data → M2 vision → M3 NLP → M4 fusion → M5 honesty + demo → M6 integration.
Each module has a "done when" check in docs/MDRIS_CONTEXT.md §14.1. Show the evidence before committing.

## Commands
- Backend: `cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Frontend: `cd frontend && npm run dev` (type check: `npm run build`)
- Tests: `cd backend && pytest -q`
- Setup check: `python check_setup.py`

## Single source of truth for constants
`backend/ai/fusion/constants.py` (fusion) and `training/common.py` (classes, urgency map, seed).
If one changes, keep the other consistent in the same commit.

## When unsure
Re-read the relevant section of docs/MDRIS_CONTEXT.md. If it does not answer the question, ask. Never invent.
