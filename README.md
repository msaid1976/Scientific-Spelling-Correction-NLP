# Scientific Spelling Correction System

A Streamlit-based scientific spelling checker that combines edit-distance candidate generation with n-gram language modeling (unigram, bigram, trigram) trained on arXiv-style corpus data.

## Key Features

- Corpus lifecycle management from the sidebar:
  - Build / rebuild preprocessed language-model artifacts
  - Manual button-triggered corpus loading
  - Clear generated artifacts
  - Live progress feedback during build/load
- Context-aware spell checking:
  - Non-word error detection (out-of-vocabulary)
  - Real-word confusable detection (e.g., their/there style patterns)
  - Ranked correction suggestions with detailed scoring signals
- Integrated analysis and visualization:
  - Spell-check tab with sample scientific texts (10 domain-focused samples)
  - Error analysis metrics and charts after each check
  - Corpus statistics and word analysis dashboards
- Vocabulary exploration:
  - Search vocabulary terms
  - Sort and paginate by frequency/alphabetic order

## Project Structure

- `SpellingCorrection.py`: Main Streamlit application and UI
- `preprocess_corpus.py`: Corpus preprocessing pipeline (builds n-gram JSON artifacts)
- `preprocessed_data/`: Generated artifacts:
  - `unigrams.json`
  - `bigrams.json`
  - `trigrams.json`
  - `vocab.json`
  - `metadata.json`
- `corpus/`: Raw corpus folder (expected file: `arxiv-metadata-oai-snapshot.json`)
- `corpus.zip`: Optional archive source for corpus extraction

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run SpellingCorrection.py
```

## Recommended Workflow

1. Open the sidebar `Corpus Management`.
2. Ensure preprocessed artifacts exist in `preprocessed_data/` (or generate them with `python preprocess_corpus.py` after preparing `corpus/` input).
3. Click `Load Scientific Corpus` to load the language model into memory.
4. Use `Rebuild Corpus` panel when you want to regenerate artifacts with new preprocessing settings.
5. Use `Spell Check` tab:
   - Choose a sample or paste your own scientific text
   - Click `Check Spelling`
   - Review:
     - `Spelling Errors`
     - `Error Analysis`
     - `Confusable Words`
6. Use `Show Corpus Statistics` for corpus-level visual analytics.

## Core Modeling Logic

- Tokenization: regex-based alphabetic token extraction
- Candidate generation:
  - Known word
  - Edit distance 1
  - Edit distance 2
- Ranking signals:
  - Unigram probability
  - Bigram context probability with configurable add-k smoothing
  - Optional trigram support
  - Confusable-word prior weights
- Preprocessing controls:
  - `target_words`
  - `min_word_frequency`
  - `bigram_smoothing_k`

## Notes

- This project focuses on scientific-domain spelling and context correction, not general grammar checking.
- Corpus quality and domain coverage directly affect correction quality.
- Very rare scientific terms may be filtered when `min_word_frequency` is set too high.

## Development

Quick syntax validation:

```bash
python -m py_compile SpellingCorrection.py preprocess_corpus.py
```

## License

Add your preferred license (e.g., MIT) if you plan to make the repository public for reuse.
