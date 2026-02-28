# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: Streamlit UI and spelling-correction logic (tokenization, edit distance, n-gram scoring, confusable-word handling).
- `preprocess_corpus.py`: Offline corpus pipeline that builds n-gram artifacts.
- `corpus/`: Raw source data (`arxiv-metadata-oai-snapshot.json`).
- `preprocessed_data/`: Generated model artifacts (`unigrams.json`, `bigrams.json`, `trigrams.json`, `vocab.json`, `metadata.json`).
- `requirements.txt`: Runtime dependencies.
- `sample.txt`: Small text sample for manual checks.

## Build, Test, and Development Commands
- `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`: create and activate the virtual environment.
- `pip install -r requirements.txt`: install Streamlit and NLP dependencies.
- `python preprocess_corpus.py`: generate language-model files from the corpus into `preprocessed_data/`.
- `streamlit run main.py`: run the local web app.
- `python -m py_compile main.py preprocess_corpus.py`: quick syntax validation.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and clear docstrings for public functions/classes.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants (for example, `CONFUSABLES`).
- Keep UI code grouped in `main()` and keep reusable NLP logic in methods/functions to avoid tightly coupled Streamlit callbacks.

## Testing Guidelines
- There is no committed automated test suite yet; use manual smoke tests before PRs.
- Minimum check: run `python -m py_compile ...`, then launch `streamlit run main.py` and verify spelling and confusable-word suggestions on `sample.txt`.
- When adding tests, place them in `tests/` and use `test_<module>.py` naming with `pytest`.

## Commit & Pull Request Guidelines
- Git metadata is not included in this workspace snapshot, so no historical commit convention is available to infer.
- Use concise, imperative commit messages; prefer Conventional Commit style (for example, `feat: improve trigram scoring`).
- PRs should include: purpose, key changes, validation steps/commands run, and screenshots for UI-impacting updates.

## Data & Configuration Notes
- Do not commit virtualenv folders, generated artifacts, or local cache files (already covered by `.gitignore`).
- Keep large raw datasets out of PR diffs unless the change explicitly requires dataset updates.
