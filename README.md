# Scientific Spelling Correction System

A Streamlit application for scientific-domain spelling correction using edit-distance candidate generation, unigram and bigram language modeling, and context-aware ranking built from an arXiv-based corpus.

## Overview

This project focuses on spelling correction in scientific writing rather than generic everyday text. It supports:

- corpus preprocessing and rebuild workflows
- loading a scientific language model into memory
- non-word and real-word/confusable error detection
- ranked correction suggestions with context scores
- corpus visualization and vocabulary exploration

## Streamlit App Preview

### 1. Corpus Management

The sidebar manages the full corpus lifecycle, from preprocessing configuration to loading and rebuilding the language model.

<p align="center">
  <img src="assets/03-Corpus Management.jpg" alt="Corpus Management" width="900">
</p>

<p align="center">
  <img src="assets/005- Complete Corpus Lifecycle Control Interface.jpg" alt="Corpus Lifecycle Control Interface" width="900">
</p>

### 2. Spell Check Workspace

Users can paste their own scientific text or load predefined domain-focused samples before running the spell checker.

<p align="center">
  <img src="assets/008- Spell Checker tab.jpg" alt="Spell Checker Tab" width="900">
</p>

<p align="center">
  <img src="assets/009- Spell Check - Predefined Samples.jpg" alt="Predefined Samples" width="900">
</p>

### 3. Correction Results and Analysis

The app highlights errors, ranks correction candidates, and explains results through tables and visual analysis.

<p align="center">
  <img src="assets/011- spell check-Results.jpg" alt="Spell Check Results" width="900">
</p>

<p align="center">
  <img src="assets/018- Suggested Corrections Tab 2- Error Analysis 1.jpg" alt="Error Analysis" width="900">
</p>

### 4. Word Explorer

The vocabulary browser supports search, sorting, pagination, and quick inspection of word probabilities and frequencies.

<p align="center">
  <img src="assets/012- Word Explorer Vocabulary Interface.jpg" alt="Word Explorer" width="900">
</p>

### 5. Corpus Visualizations

The visualization module helps inspect corpus scale, frequency behavior, and word-level patterns.

<p align="center">
  <img src="assets/007-Corpus Visualization - Corpus Statistics.jpg" alt="Corpus Statistics" width="900">
</p>

<p align="center">
  <img src="assets/07-Corpus Visualization - Word Analysis.jpg" alt="Word Analysis" width="900">
</p>

## Key Features

- Scientific-domain spelling correction tailored for research-style text
- Corpus build and rebuild tools directly inside the Streamlit sidebar
- Non-word error detection for out-of-vocabulary terms
- Confusable-word detection for context-sensitive mistakes
- Ranked candidate suggestions using:
  - unigram frequency
  - bigram context probability
  - edit distance
  - confusable-word weighting
- Error analysis dashboard with metrics and charts
- Word Explorer for browsing vocabulary entries
- Corpus statistics and word analysis visualizations

## Dataset Source

The scientific corpus used in this project comes from the arXiv dataset published on Kaggle:

- [Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)

The preprocessing pipeline expects arXiv metadata content and builds language-model artifacts from that scientific source.

## Repository Structure

```text
Scientific-Spelling-Correction-NLP/

- 📁 assets/
  screenshots used in this README and Streamlit app previews

- 📁 components/
  - 📁 interactive_text_menu/
    - 📄 index.html

- 📁 corpus/
  - 📄 arxiv-metadata-oai-snapshot.json  # expected raw corpus input

- 📁 preprocessed_data/
  - 📄 unigrams.json
  - 📄 bigrams.json
  - 📄 trigrams.json
  - 📄 vocab.json
  - 📄 metadata.json

- 📄 SpellingCorrection.py
- 📄 preprocess_corpus.py
- 📄 requirements.txt
- 📄 sample.txt
- 📄 README.md
```

## How It Works

### Preprocessing

`preprocess_corpus.py` prepares the scientific corpus into reusable artifacts:

- unigram counts
- bigram counts
- trigram counts
- vocabulary data
- metadata for corpus statistics

### Spelling Correction Logic

`SpellingCorrection.py` powers the Streamlit interface and correction engine:

- tokenizes scientific text
- detects likely spelling and confusable-word errors
- generates candidates with edit distance
- ranks candidates using frequency and contextual probability
- visualizes error patterns and corpus statistics

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run SpellingCorrection.py
```

## Recommended Workflow

1. Place the arXiv source file in `corpus/`.
2. Build preprocessing artifacts with the sidebar controls or `preprocess_corpus.py`.
3. Load the scientific corpus in the Streamlit sidebar.
4. Open `Spell Check` to test sample or custom text.
5. Review `Suggested Corrections`, `Error Analysis`, and `Confusable Words`.
6. Use `Word Explorer` and the visualization panel for corpus inspection.

## Technologies

- Python
- Streamlit
- Pandas
- Altair
- N-gram language modeling

## Notes

- This project is designed for scientific writing support, not full grammar correction.
- Correction quality depends heavily on corpus quality and vocabulary coverage.
- Rare domain terms may be filtered depending on preprocessing thresholds.
