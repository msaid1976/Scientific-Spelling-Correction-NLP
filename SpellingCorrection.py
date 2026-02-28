"""
Spelling Correction System with N-gram Language Model

This module implements a comprehensive spelling correction system that combines
statistical language modeling with edit distance algorithms to detect and correct
both non-word errors and real-word confusable errors.

Key Features:
- N-gram based language modeling (unigrams, bigrams, trigrams)
- Edit distance calculation for spelling suggestions
- Context-aware correction using surrounding words
- Confusable words detection (their/there/they're, etc.)
- Interactive Streamlit web interface
"""
import streamlit as st
import json
import re
from collections import Counter, defaultdict
import math
import os
import time
import shutil
import zipfile
import altair as alt
import pandas as pd

from preprocess_corpus import preprocess_corpus

REQUIRED_PREPROCESSED_FILES = [
    "unigrams.json",
    "bigrams.json",
    "trigrams.json",
    "vocab.json",
    "metadata.json",
]


def preprocessed_data_ready(preprocessed_dir):
    """Check whether all required preprocessed artifacts exist."""
    if not os.path.isdir(preprocessed_dir):
        return False
    return all(
        os.path.exists(os.path.join(preprocessed_dir, filename))
        for filename in REQUIRED_PREPROCESSED_FILES
    )


def clear_preprocessed_data(preprocessed_dir):
    """Delete all files and folders inside the preprocessed data directory."""
    if not os.path.isdir(preprocessed_dir):
        return 0

    deleted_count = 0
    for entry in os.listdir(preprocessed_dir):
        entry_path = os.path.join(preprocessed_dir, entry)
        if os.path.isfile(entry_path) or os.path.islink(entry_path):
            os.remove(entry_path)
            deleted_count += 1
        elif os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            deleted_count += 1
    return deleted_count


def ensure_corpus_exists(corpus_dir, corpus_zip, progress_callback=None):
    """Ensure corpus JSON exists; extract corpus.zip when corpus is missing."""
    corpus_json = os.path.join(corpus_dir, "arxiv-metadata-oai-snapshot.json")
    if os.path.exists(corpus_json):
        if progress_callback:
            progress_callback(
                {
                    "stage": "corpus",
                    "progress": 0.15,
                    "message": "Existing corpus found; extraction skipped.",
                    "details": {"corpus_path": corpus_json},
                }
            )
        return corpus_json

    if not os.path.exists(corpus_zip):
        raise FileNotFoundError(f"Corpus archive not found: {corpus_zip}")

    if progress_callback:
        progress_callback(
            {
                "stage": "corpus",
                "progress": 0.03,
                "message": "Corpus folder missing. Extracting corpus.zip.",
                "details": {"archive": corpus_zip},
            }
        )

    with zipfile.ZipFile(corpus_zip, "r") as zip_ref:
        members = zip_ref.infolist()
        total_members = max(1, len(members))
        for idx, member in enumerate(members, start=1):
            zip_ref.extract(member, ".")
            if progress_callback and (idx % 10 == 0 or idx == total_members):
                progress_callback(
                    {
                        "stage": "corpus",
                        "progress": 0.03 + (0.12 * (idx / total_members)),
                        "message": "Extracting corpus archive.",
                        "details": {
                            "files_extracted": idx,
                            "total_files": total_members,
                            "current_file": member.filename,
                        },
                    }
                )

    if not os.path.exists(corpus_json):
        raise FileNotFoundError(
            f"Expected corpus file missing after extraction: {corpus_json}"
        )

    if progress_callback:
        progress_callback(
            {
                "stage": "corpus",
                "progress": 0.15,
                "message": "Corpus extraction complete.",
                "details": {"corpus_path": corpus_json},
            }
        )

    return corpus_json


def load_model_into_session(preprocessed_dir, bigram_smoothing_k=0.1):
    """Load the spelling corrector and cached vocabulary into session state."""
    st.session_state.corrector = SpellingCorrector(
        preprocessed_dir,
        bigram_smoothing_k=bigram_smoothing_k,
    )
    st.session_state.vocab_list = st.session_state.corrector.get_sorted_vocabulary()


def _build_visualization_data(corrector):
    """Build derived dataframes used by the visualization tabs."""
    sorted_vocab = [
        (word, freq)
        for word, freq in corrector.get_sorted_vocabulary()
        if word.isalpha()
    ]
    if not sorted_vocab:
        sorted_vocab = [("n/a", 1)]

    total_tokens = sum(freq for _, freq in sorted_vocab)
    vocab_size = len(sorted_vocab)

    top_terms_df = pd.DataFrame(
        sorted_vocab[:20], columns=["term", "frequency"]
    )

    rank_limit = min(3000, vocab_size)
    rank_rows = []
    for idx, (_, freq) in enumerate(sorted_vocab[:rank_limit], start=1):
        rank_rows.append({"rank": idx, "frequency": freq})
    rank_df = pd.DataFrame(rank_rows)

    cumulative_rows = []
    checkpoints = [10, 50, 100, 500, 1000, 5000, 10000]
    cumulative_sum = 0
    checkpoint_set = {v for v in checkpoints if v <= vocab_size}
    for idx, (_, freq) in enumerate(sorted_vocab, start=1):
        cumulative_sum += freq
        if idx in checkpoint_set:
            cumulative_rows.append(
                {"top_n": idx, "coverage_pct": (cumulative_sum / max(1, total_tokens)) * 100.0}
            )
    if not cumulative_rows:
        cumulative_rows.append({"top_n": vocab_size, "coverage_pct": 100.0})
    coverage_df = pd.DataFrame(cumulative_rows)

    prefix_counter = Counter()
    suffix_counter = Counter()
    initial_counter = Counter()
    length_counter = Counter()
    frequency_bands = Counter(
        {
            "1 (Hapax)": 0,
            "2-5": 0,
            "6-20": 0,
            "21-100": 0,
            ">100": 0,
        }
    )

    for word, freq in sorted_vocab:
        if len(word) >= 3:
            prefix_counter[word[:3]] += freq
            suffix_counter[word[-3:]] += freq
        initial_counter[word[0]] += freq
        length_bucket = str(len(word)) if len(word) <= 20 else "20+"
        length_counter[length_bucket] += freq

        if freq == 1:
            frequency_bands["1 (Hapax)"] += 1
        elif freq <= 5:
            frequency_bands["2-5"] += 1
        elif freq <= 20:
            frequency_bands["6-20"] += 1
        elif freq <= 100:
            frequency_bands["21-100"] += 1
        else:
            frequency_bands[">100"] += 1

    top_prefixes_df = pd.DataFrame(
        prefix_counter.most_common(15), columns=["prefix", "frequency"]
    )
    top_suffixes_df = pd.DataFrame(
        suffix_counter.most_common(15), columns=["suffix", "frequency"]
    )

    def _length_sort_key(value):
        if value == "20+":
            return 999
        return int(value)

    word_length_df = pd.DataFrame(
        [{"length": length, "count": count} for length, count in length_counter.items()]
    )
    word_length_df = word_length_df.sort_values(
        by="length", key=lambda s: s.map(_length_sort_key)
    )

    initial_letter_df = pd.DataFrame(
        initial_counter.most_common(15), columns=["letter", "frequency"]
    )

    frequency_band_df = pd.DataFrame(
        [{"band": k, "term_count": v} for k, v in frequency_bands.items()]
    )

    bigram_unique = sum(len(next_words) for next_words in corrector.bigrams.values())
    trigram_unique = 0
    for nested in corrector.trigrams.values():
        trigram_unique += sum(len(next_words) for next_words in nested.values())

    corpus_metrics = {
        "total_tokens": int(total_tokens),
        "vocab_size": int(vocab_size),
        "avg_token_frequency": float(total_tokens / max(1, vocab_size)),
        "unique_bigrams": int(bigram_unique),
        "unique_trigrams": int(trigram_unique),
    }

    return {
        "corpus_metrics": corpus_metrics,
        "top_terms_df": top_terms_df,
        "rank_df": rank_df,
        "coverage_df": coverage_df,
        "top_prefixes_df": top_prefixes_df,
        "top_suffixes_df": top_suffixes_df,
        "word_length_df": word_length_df,
        "initial_letter_df": initial_letter_df,
        "frequency_band_df": frequency_band_df,
    }


def _get_visualization_data(corrector):
    """Cache visualization data in session state for current loaded model."""
    model_key = (
        int(corrector.total_words),
        int(len(corrector.vocab)),
        int(getattr(corrector, "min_word_frequency", 1)),
    )
    cached = st.session_state.get("viz_data_cache")
    if cached and cached.get("key") == model_key:
        return cached["data"]

    data = _build_visualization_data(corrector)
    st.session_state["viz_data_cache"] = {"key": model_key, "data": data}
    return data


def render_visualizations_panel(corrector):
    """Render 3-tab corpus visualization panel."""
    viz_data = _get_visualization_data(corrector)
    corpus_metrics = viz_data["corpus_metrics"]
    top_terms_df = viz_data["top_terms_df"]
    rank_df = viz_data["rank_df"]
    coverage_df = viz_data["coverage_df"]
    top_prefixes_df = viz_data["top_prefixes_df"]
    top_suffixes_df = viz_data["top_suffixes_df"]
    word_length_df = viz_data["word_length_df"]
    initial_letter_df = viz_data["initial_letter_df"]
    frequency_band_df = viz_data["frequency_band_df"]

    st.markdown("## 📈 Corpus Visualizations")
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(
        ["📊 Corpus Statistics", "🔤 Word Analysis", "❌ Error Analysis"]
    )

    with viz_tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tokens", f"{corpus_metrics['total_tokens']:,}")
        m2.metric("Vocabulary Size", f"{corpus_metrics['vocab_size']:,}")
        m3.metric("Unique Bigrams", f"{corpus_metrics['unique_bigrams']:,}")
        m4.metric("Unique Trigrams", f"{corpus_metrics['unique_trigrams']:,}")

        col1, col2 = st.columns(2)
        with col1:
            top_terms_chart = (
                alt.Chart(top_terms_df)
                .mark_bar(color="#e89a7d")
                .encode(
                    x=alt.X("frequency:Q", title="Frequency"),
                    y=alt.Y("term:N", sort="-x", title="Scientific Terms"),
                    tooltip=["term:N", "frequency:Q"],
                )
                .properties(title="Top 20 Most Frequent Scientific Terms", height=360)
            )
            st.altair_chart(top_terms_chart, width="stretch")
        with col2:
            zipf_chart = (
                alt.Chart(rank_df)
                .mark_line(color="#4f79b5")
                .encode(
                    x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="Word Rank (log scale)"),
                    y=alt.Y("frequency:Q", scale=alt.Scale(type="log"), title="Frequency (log scale)"),
                    tooltip=["rank:Q", "frequency:Q"],
                )
                .properties(title="Zipf Curve (Rank vs Frequency)", height=360)
            )
            st.altair_chart(zipf_chart, width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            coverage_chart = (
                alt.Chart(coverage_df)
                .mark_line(point=True, color="#2a9d8f")
                .encode(
                    x=alt.X("top_n:Q", title="Top-N Frequent Terms"),
                    y=alt.Y("coverage_pct:Q", title="Token Coverage (%)"),
                    tooltip=[
                        alt.Tooltip("top_n:Q", title="Top-N"),
                        alt.Tooltip("coverage_pct:Q", title="Coverage", format=".2f"),
                    ],
                )
                .properties(title="Cumulative Token Coverage by Top-N Terms", height=300)
            )
            st.altair_chart(coverage_chart, width="stretch")
        with col4:
            band_order = ["1 (Hapax)", "2-5", "6-20", "21-100", ">100"]
            band_chart = (
                alt.Chart(frequency_band_df)
                .mark_bar(color="#b089c6")
                .encode(
                    x=alt.X("band:N", sort=band_order, title="Frequency Band"),
                    y=alt.Y("term_count:Q", title="Number of Unique Terms"),
                    tooltip=["band:N", "term_count:Q"],
                )
                .properties(title="Vocabulary Frequency Bands", height=300)
            )
            st.altair_chart(band_chart, width="stretch")

    with viz_tab2:
        col1, col2 = st.columns(2)
        with col1:
            length_chart = (
                alt.Chart(word_length_df)
                .mark_bar(color="#7fa8cc")
                .encode(
                    x=alt.X("length:O", title="Word Length (characters)"),
                    y=alt.Y("count:Q", title="Number of Words"),
                    tooltip=["length:O", "count:Q"],
                )
                .properties(title="Scientific Term Length Distribution", height=340)
            )
            st.altair_chart(length_chart, width="stretch")

        with col2:
            prefix_chart = (
                alt.Chart(top_prefixes_df)
                .mark_bar(color="#99d8a0")
                .encode(
                    x=alt.X("frequency:Q", title="Frequency"),
                    y=alt.Y("prefix:N", sort="-x", title="Prefix (3 letters)"),
                    tooltip=["prefix:N", "frequency:Q"],
                )
                .properties(title="Most Common Scientific Prefixes", height=340)
            )
            st.altair_chart(prefix_chart, width="stretch")

        col3, col4 = st.columns(2)
        with col3:
            suffix_chart = (
                alt.Chart(top_suffixes_df)
                .mark_bar(color="#f2b36f")
                .encode(
                    x=alt.X("frequency:Q", title="Frequency"),
                    y=alt.Y("suffix:N", sort="-x", title="Suffix (3 letters)"),
                    tooltip=["suffix:N", "frequency:Q"],
                )
                .properties(title="Most Common Scientific Suffixes", height=300)
            )
            st.altair_chart(suffix_chart, width="stretch")
        with col4:
            letter_chart = (
                alt.Chart(initial_letter_df)
                .mark_bar(color="#7cb8b2")
                .encode(
                    x=alt.X("letter:N", sort="-y", title="Initial Letter"),
                    y=alt.Y("frequency:Q", title="Frequency"),
                    tooltip=["letter:N", "frequency:Q"],
                )
                .properties(title="Top Initial Letters in Scientific Terms", height=300)
            )
            st.altair_chart(letter_chart, width="stretch")

    with viz_tab3:
        error_info = st.session_state.get("error_info", {})
        total_errors = len(error_info)
        total_words_checked = int(st.session_state.get("last_checked_word_count", 0))
        homophone_errors = 0
        for _, value in error_info.items():
            word, _, _, _ = value
            if word.lower() in CONFUSABLES:
                homophone_errors += 1
        non_word_errors = sum(
            1 for _, _, _, error_type in error_info.values() if error_type == "non-word"
        )
        context_errors = sum(
            1 for _, _, _, error_type in error_info.values() if error_type == "real-word"
        )

        error_df = pd.DataFrame(
            [
                {"type": "Non-word errors", "count": non_word_errors},
                {"type": "Context errors", "count": context_errors},
                {"type": "Homophone errors", "count": homophone_errors},
            ]
        )
        error_rate = (
            (total_errors / total_words_checked) if total_words_checked > 0 else 0.0
        )
        donut_df = pd.DataFrame(
            [
                {"segment": "Error", "value": error_rate},
                {"segment": "Clean", "value": max(0.0, 1.0 - error_rate)},
            ]
        )

        col1, col2 = st.columns(2)
        with col1:
            error_chart = (
                alt.Chart(error_df)
                .mark_bar()
                .encode(
                    x=alt.X("type:N", title=""),
                    y=alt.Y("count:Q", title="Number of Errors"),
                    color=alt.Color(
                        "type:N",
                        scale=alt.Scale(
                            domain=["Non-word errors", "Context errors", "Homophone errors"],
                            range=["#e9858a", "#76c0c1", "#c8c8c8"],
                        ),
                        legend=None,
                    ),
                    tooltip=["type:N", "count:Q"],
                )
                .properties(title="Error Type Distribution", height=420)
            )
            st.altair_chart(error_chart, width="stretch")

        with col2:
            donut_chart = (
                alt.Chart(donut_df)
                .mark_arc(innerRadius=110)
                .encode(
                    theta=alt.Theta("value:Q"),
                    color=alt.Color(
                        "segment:N",
                        scale=alt.Scale(domain=["Error", "Clean"], range=["#fb696e", "#d9d9d9"]),
                        legend=None,
                    ),
                    tooltip=["segment:N", alt.Tooltip("value:Q", format=".1%")],
                )
                .properties(title="Text Error Rate", height=420)
            )
            center_text_df = pd.DataFrame(
                [{"label": f"{error_rate * 100:.1f}%\nError Rate", "x": 0, "y": 0}]
            )
            center_text = (
                alt.Chart(center_text_df)
                .mark_text(fontSize=26, fontWeight=700, align="center")
                .encode(
                    x=alt.value(205),
                    y=alt.value(205),
                    text="label:N",
                )
            )
            st.altair_chart(donut_chart + center_text, width="stretch")

        if total_words_checked == 0:
            st.info("Run a spell check to populate live Error Analysis charts.")

# Edit distance functions
def min_edit_distance(source, target, ins_cost=1, del_cost=1, sub_cost=1):
    """
    Calculate the minimum edit distance between two words using dynamic programming.
    
    The edit distance (Levenshtein distance) is the minimum number of single-character 
    edits required to change one word into another.
    
    Args:
        source (str): The source word to transform
        target (str): The target word to match
        ins_cost (int, optional): Cost of inserting a character. Defaults to 1.
        del_cost (int, optional): Cost of deleting a character. Defaults to 1.
        sub_cost (int, optional): Cost of substituting a character. Defaults to 1.
    
    Returns:
        int: The minimum edit distance between source and target
    """
    n = len(source) # Lenght of source word
    m = len(target) # Lenght of target word
    
    # Create a distance matrix
    # D[i][j] represents the edit distance between source[:i] and target[:j]
    D = [[0 for _ in range(m+1)] for _ in range(n+1)]
    
    # Initialize the first row
    # This requires j insertions
    for i in range(1, n+1):
        D[i][0] = D[i-1][0] + del_cost
        
    # Initialize the first column
    # this requires i deletions
    for j in range(1, m+1):
        D[0][j] = D[0][j-1] + ins_cost
        
    # Fill the matrix 
    for i in range(1, n+1):
        for j in range(1, m+1):
            # if characters match, no additional cost
            if source[i-1] == target[j-1]:
                d = 0
            else:
                d = sub_cost
                
            # Choose the minimum cost operation
            D[i][j] = min(
                D[i-1][j] + del_cost,
                D[i][j-1] + ins_cost,
                D[i-1][j-1] + d
            )
    
    # Return the edit distance between the full words
    return D[n][m]

# Enhanced confusables dictionary with weights
# Each entry maps a word to a list of confused word tuples
CONFUSABLES = {
    # common homophone confusions
    "their": [("there", 0.9), ("they're", 0.8)],
    "there": [("their", 0.9), ("they're", 0.7)],
    "they're": [("their", 0.8), ("there", 0.7)],

    # Possessive vs contraction confusions
    "your": [("you're", 0.9)],
    "you're": [("your", 0.9)],
    "its": [("it's", 0.9)],
    "it's": [("its", 0.9)],

    # temporal / comparative confusions
    "then": [("than", 0.8)],
    "than": [("then", 0.8)],

    # Noun / verb confusions
    "affect": [("effect", 0.85)],
    "effect": [("affect", 0.85)],

    # Perposition confusions
    "accept": [("except", 0.8)],
    "except": [("accept", 0.8)],

    # Condition/weather confusions
    "weather": [("whether", 0.8)],
    "whether": [("weather", 0.8)],

    # Verb form confusions
    "loose": [("lose", 0.9)],
    "lose": [("loose", 0.9)],

    # Similar meaning word confusions
    "principal": [("principle", 0.85)],
    "principle": [("principal", 0.85)],
    "advice": [("advise", 0.8)],
    "advise": [("advice", 0.8)],

    # British / american spelling variants
    "practice": [("practise", 0.7)],
    "practise": [("practice", 0.7)],
}

# Spelling correction system class
class SpellingCorrector:
    """
    A comprehensive spelling correction system using n-gram language models.
    
    This class implements a statistical approach to spelling correction that combines:
    - N-gram language modeling for context awareness
    - Edit distance algorithms for generating correction candidates
    - Confusable words detection for real-word errors
    - Probabilistic scoring for ranking suggestions
    
    Attributes:
        unigrams (Counter): Word frequency counts from the corpus
        bigrams (defaultdict): Bigram frequency counts (word pairs)
        trigrams (defaultdict): Trigram frequency counts (word triples)
        vocab (set): Set of all valid words in the vocabulary
        total_words (int): Total number of words processed from corpus
        confusables (dict): Dictionary of commonly confused word pairs
    """
    def __init__(self, preprocessed_dir="preprocessed_data", bigram_smoothing_k=0.1):
        """
        Initialize the spelling corrector with preprocessed language model data.
        
        Args:
            preprocessed_dir (str): Directory containing preprocessed n-gram data files
            bigram_smoothing_k (float): Add-k smoothing value for bigram probabilities
        """

        # Initialise data structures for model
        self.unigrams = Counter()   # Single word frequencies
        self.bigrams = defaultdict(Counter) # Word Pair frequencies
        self.trigrams = defaultdict(lambda: defaultdict(Counter))   # Word Triple frequencies
        self.vocab = set()  #   Valid vocabulary set
        self.total_words = 0    # Total Corpus size
        self.confusables = CONFUSABLES  # Confusable words dictionary
        self.bigram_smoothing_k = float(bigram_smoothing_k)
        self.vocab_size = 0
        self.min_word_frequency = 1

        # Load the preprocessed language model data
        self.load_preprocessed_data(preprocessed_dir)
        
    def load_preprocessed_data(self, preprocessed_dir):
        """
        Load preprocessed n-gram data from JSON files.
        
        Expected files in preprocessed_dir:
        - unigrams.json: Word frequency counts
        - bigrams.json: Bigram frequency counts
        - trigrams.json: Trigram frequency counts
        - vocab.json: List of valid vocabulary words
        - metadata.json: Corpus statistics
        
        Args:
            preprocessed_dir (str): Directory path containing preprocessed data files
            
        Raises:
            FileNotFoundError: If preprocessed directory or required files don't exist
            JSONDecodeError: If JSON files are corrupted or malformed
        """
        st.info("Loading preprocessed data...")
        
        # Check if preprocessed data exists
        if not os.path.exists(preprocessed_dir):
            st.error(f"Preprocessed data directory '{preprocessed_dir}' not found.")
            st.info("Please run the preprocess_corpus.py script first.")
            return
        
        try:
            # Load unigrams
            with open(os.path.join(preprocessed_dir, "unigrams.json"), 'r') as f:
                self.unigrams = Counter(json.load(f))
            
            # Load bigrams
            with open(os.path.join(preprocessed_dir, "bigrams.json"), 'r') as f:
                bigrams_data = json.load(f)
                self.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in bigrams_data.items()})
            
            # Load trigrams
            with open(os.path.join(preprocessed_dir, "trigrams.json"), 'r') as f:
                trigrams_data = json.load(f)
                self.trigrams = defaultdict(lambda: defaultdict(Counter))
                for w1, nested in trigrams_data.items():
                    for w2, subnested in nested.items():
                        self.trigrams[w1][w2] = Counter(subnested)
            
            # Load vocabulary
            with open(os.path.join(preprocessed_dir, "vocab.json"), 'r') as f:
                self.vocab = set(json.load(f))
            self.vocab_size = len(self.vocab)
            
            # Load metadata
            with open(os.path.join(preprocessed_dir, "metadata.json"), 'r') as f:
                metadata = json.load(f)
                self.total_words = int(metadata.get("total_words", sum(self.unigrams.values())))
                self.min_word_frequency = int(metadata.get("min_word_frequency", 1))
            
            st.success(
                f"Preprocessed data loaded with {self.total_words} words and "
                f"{len(self.vocab)} unique words (min_freq={self.min_word_frequency}, "
                f"k={self.bigram_smoothing_k})."
            )
            
        except Exception as e:
            st.error(f"Error loading preprocessed data: {e}")
            st.info("Please run the preprocess_corpus.py script first.")
    
    def tokenize(self, text):
        """
        Tokenize input text into individual words.
        
        Uses regex to extract alphabetic words while ignoring punctuation and numbers.
        All words are converted to lowercase for consistency.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of lowercase word tokens
        """

        # Extract only alphabetic word sequences and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def P(self, word):
        """
        Calculate the unigram probability of a word.
        
        Uses maximum likelihood estimation: P(word) = count(word) / total_words
        
        Args:
            word (str): Word to calculate probability for
            
        Returns:
            float: Probability of the word (0.0 if word not in corpus or corpus empty)
        """
        return self.unigrams[word] / self.total_words if self.total_words > 0 else 0
    
    def conditional_probability(self, word1, word2):
        """
        Calculate the conditional probability P(word2|word1) using bigrams.
        
        Uses add-k smoothing when k>0:
        P(word2|word1) = (count(word1,word2)+k) / (count(word1)+k*|V|)
        
        Args:
            word1 (str): Previous word (condition)
            word2 (str): Current word (target)
            
        Returns:
            float: Conditional probability of word2 given word1
        """
        count_w1 = self.unigrams[word1]
        vocab_size = max(1, self.vocab_size)
        k = max(0.0, float(self.bigram_smoothing_k))

        if count_w1 <= 0:
            return (1.0 / vocab_size) if k > 0 else 0.0

        if k > 0:
            return (self.bigrams[word1][word2] + k) / (count_w1 + (k * vocab_size))

        if word1 in self.bigrams and self.bigrams[word1][word2] > 0:
            return self.bigrams[word1][word2] / count_w1
        return 0.0
    
    def trigram_probability(self, word1, word2, word3):
        """
        Calculate the conditional probability P(word3|word1,word2) using trigrams.
        
        Uses maximum likelihood estimation: P(word3|word1,word2) = count(word1,word2,word3) / count(word1,word2)
        
        Args:
            word1 (str): First previous word
            word2 (str): Second previous word  
            word3 (str): Current word (target)
            
        Returns:
            float: Conditional probability of word3 given word1 and word2
        """
        # Check if trigram data is available
        if (word1 in self.trigrams and word2 in self.trigrams[word1] and 
            self.trigrams[word1][word2][word3] > 0 and sum(self.trigrams[word1][word2].values()) > 0):
            return self.trigrams[word1][word2][word3] / sum(self.trigrams[word1][word2].values())
        return 0
    
    def known(self, words):
        """
        Filter a list of words to return only those in the vocabulary.
        
        Args:
            words (List[str]): List of candidate words to filter
            
        Returns:
            Set[str]: Subset of words that exist in the vocabulary
        """

        return set(w for w in words if w in self.vocab)
    
    def edits1(self, word):
        """
        Generate all possible words that are one edit away from the input word.
        
        Args:
            word (str): Input word to generate edits for
            
        Returns:
            Set[str]: Set of all words one edit away from input
        """

        letters = 'abcdefghijklmnopqrstuvwxyz'

        # Generate all possible split points in the word
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        # Deletions
        deletes = [L + R[1:] for L, R in splits if R]

        # Transpositions
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

        # Replacements
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

        # Insertions
        inserts = [L + c + R for L, R in splits for c in letters]

        # Return all unique edits
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word):
        """
        Generate all possible words that are two edits away from the input word.
        
        Args:
            word (str): Input word to generate edits for
            
        Returns:
            Set[str]: Set of all words two edits away from input
        """

        # Apply edits1 to each result of edits1 
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    
    def candidates(self, word):
        """
        Generate candidate corrections for a potentially misspelled word.
        
        Args:
            word (str): Potentially misspelled word
            
        Returns:
            List[str]: List of candidate corrections in priority order
        """
        # First check if it's a known confusable
        if word in self.confusables:
            return [cand for cand, _ in self.confusables[word]] + [word]
        
        # Generate candidate corrections
        candidates = (self.known([word]) or 
                      self.known(self.edits1(word)) or 
                      self.known(self.edits2(word)) or 
                      [word])
        
        # Filter out very short candidates (1-2 letters) unless they're the original word
        if len(word) > 2:
            candidates = [c for c in candidates if len(c) > 1 or c == word]
        
        return candidates
    
    def correction(self, word, prev_word=None, next_word=None):
        """
         Find the most probable spelling correction for a word using context.
        
        Uses a combination of:
        - Unigram probabilities (word frequency)
        - Bigram probabilities (context with adjacent words)
        - Trigram probabilities (context with both adjacent words)
        - Confusable word weights
        
        Args:
            word (str): Word to correct
            prev_word (Optional[str]): Previous word for context
            next_word (Optional[str]): Next word for context
            
        Returns:
            str: Most probable correction
        """
        candidates = list(self.candidates(word))
        
        # If no context, return the most probable candidate
        if not prev_word and not next_word:
            return max(candidates, key=self.P)
        
        # Score candidates based on context
        scored_candidates = []
        for cand in candidates:
            # Base probability score
            base_score = math.log(self.P(cand) + 1e-10) if self.P(cand) > 0 else -100
            
            # Context score bigram probability with previous word
            context_score = 0
            if prev_word:
                context_score += math.log(self.conditional_probability(prev_word, cand) + 1e-10)
            
            # Context score bigram probability with next word
            if next_word:
                context_score += math.log(self.conditional_probability(cand, next_word) + 1e-10)
            
            # If we have both previous and next words, use trigram probability
            if prev_word and next_word:
                trigram_score = math.log(self.trigram_probability(prev_word, cand, next_word) + 1e-10)
                context_score += 0.5 * trigram_score
            
            # Boost score for condusable words baded on confusion weight
            if word in self.confusables:
                for conf_cand, weight in self.confusables[word]:
                    if cand == conf_cand:
                        context_score += math.log(weight + 1e-10)
            
            # Combine scores
            total_score = base_score + context_score
            scored_candidates.append((cand, total_score))
        
        # Return best candidate
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[0][0]
        
        # Fallback
        return word
    
    def detect_errors(self, words):
        """
        Detect both non-word errors and real-word errors in a list of words.
        
        Two types of errors are detected:
        1. Non-word errors: Words not found in the vocabulary
        2. Real-word errors: Valid words that are commonly confused with others
        
        Args:
            words (List[str]): List of words to check for errors
            
        Returns:
            List[Tuple[int, str]]: List of (word_index, error_type) tuples
        """
        errors = []
        
        for i, word in enumerate(words):
            # Skip punctuation and very short words
            if not word.isalpha() or len(word) <= 1:
                continue
                
            word_lower = word.lower()
            
            # Check for non-word errors (not in vocabulary)
            if word_lower not in self.vocab:
                errors.append((i, "non-word"))
                continue
                
            # Check for confusables (real-word errors)
            if word_lower in self.confusables:
                errors.append((i, "real-word"))
        
        return errors
    
    def suggest_corrections_with_stats(self, word, prev_word=None, next_word=None, max_suggestions=5):
        """
        Generate detailed correction suggestions with statistical information.
        
        For each candidate correction, provides:
        - The suggested word
        - Overall ranking score
        - Corpus frequency
        - Edit distance from original
        - Context probabilities
        - Confusable word weight (if applicable)
        
        Args:
            word (str): Word to generate suggestions for
            prev_word (Optional[str]): Previous word for context
            next_word (Optional[str]): Next word for context
            max_suggestions (int): Maximum number of suggestions to return
            
        Returns:
            List[Dict]: List of suggestion dictionaries with detailed statistics
        """
        candidates = list(self.candidates(word))
        
        # Score candidates
        scored_candidates = []
        for cand in candidates:
            # Calculate edit distance
            edit_dist = min_edit_distance(word, cand)
            
            # Base probability score
            base_score = math.log(self.P(cand) + 1e-10) if self.P(cand) > 0 else -100
            frequency = self.unigrams[cand] if cand in self.unigrams else 0
            
            # Context score
            context_score = 0
            prev_prob = 0
            next_prob = 0
            
            if prev_word:
                prev_prob = self.conditional_probability(prev_word, cand)
                context_score += math.log(prev_prob + 1e-10)
            
            if next_word:
                next_prob = self.conditional_probability(cand, next_word)
                context_score += math.log(next_prob + 1e-10)
            
            # confusables weight
            confusable_weight = 0
            if word in self.confusables:
                for conf_cand, weight in self.confusables[word]:
                    if cand == conf_cand:
                        confusable_weight = weight
                        context_score += math.log(weight + 1e-10)
            
            # Combine scores
            total_score = base_score + context_score
            
            scored_candidates.append({
                'candidate': cand,
                'score': total_score,
                'frequency': frequency,
                'edit_distance': edit_dist,
                'prev_prob': prev_prob,
                'next_prob': next_prob,
                'confusable_weight': confusable_weight
            })
        
        # Sort by score and get top candidates
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return scored_candidates[:max_suggestions]
    
    def get_sorted_vocabulary(self):
        """
        Get the complete vocabulary sorted by word frequency.
        
        Returns:
            List[Tuple[str, int]]: List of (word, frequency) tuples sorted by frequency
        """
        return sorted([(word, freq) for word, freq in self.unigrams.items()], 
                     key=lambda x: x[1], reverse=True)
    
    def search_vocabulary(self, query):
        """
        Search for words in vocabulary that contain the query string.
        
        Args:
            query (str): Search term to look for within words
            
        Returns:
            List[Tuple[str, int]]: List of matching (word, frequency) tuples sorted by frequency
        """
        query = query.lower()
        results = []
        for word, freq in self.unigrams.items():
            if query in word.lower():
                results.append((word, freq))
        return sorted(results, key=lambda x: x[1], reverse=True)

# Streamlit app
def main():
    """
    Main Streamlit application function.
    
    Creates a multi-tab interface for:
    1. Spell checking with error detection and correction suggestions
    2. Vocabulary exploration and search
    3. Information about the system
    """
    st.set_page_config(page_title="Scientific Spelling Correction", layout="wide")
    
    st.title("Scientific Spelling Correction System")

    preprocessed_dir = "preprocessed_data"
    corpus_dir = "corpus"
    corpus_zip = "corpus.zip"
    metadata_path = os.path.join(preprocessed_dir, "metadata.json")
    corpus_ready = preprocessed_data_ready(preprocessed_dir)

    if "target_corpus_words" not in st.session_state:
        st.session_state.target_corpus_words = 250000
    if "min_word_freq" not in st.session_state:
        st.session_state.min_word_freq = 2
    if "bigram_smoothing_k" not in st.session_state:
        st.session_state.bigram_smoothing_k = 0.1
    if "show_visualizations" not in st.session_state:
        st.session_state.show_visualizations = False

    post_build_notice = st.session_state.pop("post_build_notice", None)
    if post_build_notice:
        st.success(post_build_notice)

    with st.sidebar:
        st.header("Corpus Management")
        panel_title = "🧱 Rebuild Corpus (Optional)" if corpus_ready else "🧱 Build Corpus"
        panel_header_color = "#0b2e59" if corpus_ready else "#b42318"
        panel_body_color = "#e8eff8" if corpus_ready else "#fdecec"
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] div[data-testid="stExpander"] > details {{
                border: 1px solid {panel_header_color};
                border-radius: 12px;
                overflow: hidden;
            }}
            [data-testid="stSidebar"] div[data-testid="stExpander"] > details > summary {{
                background: {panel_header_color};
                color: #ffffff;
                border-radius: 0;
            }}
            [data-testid="stSidebar"] div[data-testid="stExpander"] > details > summary p {{
                color: #ffffff;
                font-weight: 600;
            }}
            [data-testid="stSidebar"] div[data-testid="stExpander"] > details > div {{
                background: {panel_body_color};
                padding-top: 0.25rem;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        build_section = st.expander(panel_title, expanded=not corpus_ready)

        with build_section:
            target_corpus_words = st.number_input(
                "Target corpus words (>=100,000)",
                min_value=100000,
                step=50000,
                key="target_corpus_words",
                help="Preprocessing stops once this minimum word count is reached.",
            )
            min_word_freq = st.number_input(
                "Min word frequency in dictionary",
                min_value=1,
                step=1,
                key="min_word_freq",
                help="Words below this corpus count are pruned from vocab and n-gram tables.",
            )
            if corpus_ready:
                bigram_smoothing_k = st.selectbox(
                    "Bigram smoothing k",
                    options=[0.01, 0.05, 0.1, 0.5, 1.0],
                    key="bigram_smoothing_k",
                    help="Add-k smoothing used in P(word2|word1) bigram probabilities.",
                )
            else:
                bigram_smoothing_k = float(st.session_state.get("bigram_smoothing_k", 0.1))
                st.caption(
                    "Bigram smoothing will be applied after the model is built."
                )

            build_pressed = st.button(
                "Fetch/Build CS Corpus (arXiv)",
                type="primary",
                width="stretch",
            )
            clear_pressed = st.button("Clear Corpus", width="stretch")

        approx_words = 0
        current_min_freq = None
        current_target_words = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    approx_words = int(metadata.get("total_words", 0))
                    current_min_freq = metadata.get("min_word_frequency")
                    current_target_words = metadata.get("target_words")
            except Exception:
                approx_words = 0
        st.markdown(f"**Words (approx): {approx_words:,}**")
        if current_min_freq is not None:
            st.caption(
                f"Current model settings -> min_freq={current_min_freq}, "
                f"target_words={current_target_words}"
            )

        st.markdown("---")
        st.subheader("📈 Visualizations")
        if st.button(
            "Show Corpus Statistics",
            width="stretch",
            disabled=not corpus_ready,
        ):
            st.session_state.show_visualizations = True
        if not corpus_ready:
            st.caption("Build/load corpus first to enable visual analytics.")

    if clear_pressed:
        removed_entries = clear_preprocessed_data(preprocessed_dir)
        st.session_state.pop("corrector", None)
        st.session_state.pop("vocab_list", None)
        st.session_state.pop("viz_data_cache", None)
        st.session_state.pop("pending_model_reload", None)
        st.session_state.show_visualizations = False
        st.session_state.post_build_notice = (
            f"Cleared preprocessed corpus data ({removed_entries} entries removed from '{preprocessed_dir}')."
        )
        st.rerun()

    if build_pressed:
        progress_bar = st.progress(0, text="0% | Initializing corpus build...")
        status_placeholder = st.empty()
        build_details = st.expander("Fetch/Build Details", expanded=True)
        log_placeholder = build_details.empty()
        build_logs = []

        def update_build_ui(event):
            progress = max(0.0, min(1.0, float(event.get("progress", 0.0))))
            stage = str(event.get("stage", "build"))
            message = str(event.get("message", ""))
            details = event.get("details", {}) or {}

            percent = int(progress * 100)
            progress_bar.progress(percent, text=f"{percent}% | {stage}: {message}")
            status_placeholder.info(f"Stage: `{stage}` | {message}")

            timestamp = time.strftime("%H:%M:%S")
            detail_str = ""
            if details:
                detail_str = " | " + ", ".join(f"{k}={v}" for k, v in details.items())
            build_logs.append(f"[{timestamp}] {stage}: {message}{detail_str}")
            log_placeholder.code("\n".join(build_logs[-40:]), language="text")

        try:
            update_build_ui(
                {
                    "stage": "init",
                    "progress": 0.01,
                    "message": "Build request received.",
                    "details": {
                        "target_words": int(target_corpus_words),
                        "min_word_frequency": int(min_word_freq),
                        "bigram_smoothing_k": float(bigram_smoothing_k),
                    },
                }
            )

            corpus_json = ensure_corpus_exists(
                corpus_dir,
                corpus_zip,
                progress_callback=update_build_ui,
            )
            build_metadata = preprocess_corpus(
                corpus_json,
                output_dir=preprocessed_dir,
                min_words=int(target_corpus_words),
                min_word_frequency=int(min_word_freq),
                progress_callback=update_build_ui,
            )

            update_build_ui(
                {
                    "stage": "reload",
                    "progress": 0.98,
                    "message": "Build complete. Refreshing app and loading language model.",
                    "details": {"bigram_smoothing_k": float(bigram_smoothing_k)},
                }
            )
            update_build_ui(
                {
                    "stage": "complete",
                    "progress": 1.0,
                    "message": "Corpus build finished successfully.",
                    "details": build_metadata or {},
                }
            )
            st.session_state.pop("corrector", None)
            st.session_state.pop("vocab_list", None)
            st.session_state.pop("viz_data_cache", None)
            st.session_state.pending_model_reload = True
            st.session_state.post_build_notice = (
                "Corpus is ready. Loading the updated language model..."
            )
            time.sleep(0.3)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to build corpus data: {e}")

    # Initialize the spelling corrector when data is available
    force_reload_model = bool(st.session_state.pop("pending_model_reload", False))
    if preprocessed_data_ready(preprocessed_dir):
        if "corrector" not in st.session_state or force_reload_model:
            with st.spinner("Loading language model..."):
                load_model_into_session(
                    preprocessed_dir,
                    bigram_smoothing_k=float(bigram_smoothing_k),
                )
        else:
            st.session_state.corrector.bigram_smoothing_k = float(bigram_smoothing_k)
    else:
        st.warning(
            "Preprocessed data is missing. Use the sidebar to extract/build the corpus."
        )
        st.info(
            "Expected files: unigrams.json, bigrams.json, trigrams.json, vocab.json, metadata.json"
        )
        return

    corrector = st.session_state.corrector

    if st.session_state.get("show_visualizations", False):
        render_visualizations_panel(corrector)
        if st.button("Close Visualizations"):
            st.session_state.show_visualizations = False
            st.rerun()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Spell Check", "Word Explorer", "About"])
    
    with tab1:
        """
         Spell Check Tab
        
        Provides the main spelling correction interface where users can:
        - Input text for spell checking
        - View highlighted errors in the text
        - See detailed correction suggestions with statistics
        - Understand why certain corrections are recommended
        """
        st.markdown("Paste your text below to check for spelling errors.")
        
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            max_chars=1000,
            help="Type or paste your text here for spell checking",
            key="spell_check_input"
        )
        
        # Store errors in session state for click handling
        if 'errors' not in st.session_state:
            st.session_state.errors = {}
        if 'error_info' not in st.session_state:
            st.session_state.error_info = {}
        
        if st.button("Check Spelling", type="primary", key="check_spelling"):
            if text_input:
                with st.spinner("Analyzing text..."):
                    # Tokenize the text
                    words = re.findall(r"[\w']+|[.,!?;]", text_input)

                    # Track original positions of each token for highlighting
                    token_positions = []
                    pos = 0
                    for word in words:
                        start = text_input.find(word, pos)
                        end = start + len(word)
                        token_positions.append((word, start, end))
                        pos = end

                    alpha_word_count = sum(
                        1 for token, _, _ in token_positions if re.match(r"[A-Za-z']+$", token)
                    )
                    st.session_state.last_checked_word_count = alpha_word_count
                    
                    # Detect errors
                    errors = corrector.detect_errors([w for w, _, _ in token_positions])
                    st.session_state.errors = {i: err_type for i, err_type in errors}
                    
                    # Display summary statistics
                    non_word_count = sum(1 for _, err_type in errors if err_type == "non-word")
                    real_word_count = sum(1 for _, err_type in errors if err_type == "real-word")
                    total_errors = non_word_count + real_word_count
                    
                    # Create columns for stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Errors", total_errors)
                    with col2:
                        st.metric("Spelling Errors", non_word_count)
                    with col3:
                        st.metric("Confusable Words", real_word_count)
                    
                    # Display text with errors highlighted
                    st.subheader("Text Analysis")
                    html_text = ""
                    last_pos = 0
                    error_info = {}
                    
                    for i, (word, start, end) in enumerate(token_positions):
                        # Add text before this token
                        html_text += text_input[last_pos:start]
                        
                        # Check if this is an error
                        is_error = any(err_idx == i for err_idx, _ in errors)
                        
                        if is_error:
                            # This is an error word
                            error_type = next(err_type for err_idx, err_type in errors if err_idx == i)
                            error_class = "non-word" if error_type == "non-word" else "real-word"
                            # Make error words clickable
                            html_text += f'<span class="{error_class}" id="error-{i}" onclick="handleWordClick({i})">{word}</span>'
                            error_info[i] = (word, start, end, error_type)
                        else:
                            html_text += word
                        
                        last_pos = end
                    
                    # Add remaining text
                    html_text += text_input[last_pos:]
                    
                    # Store error info in session state
                    st.session_state.error_info = error_info
                    
                    # Display with custom CSS - improved contrast
                    st.markdown(f"""
                    <style>
                    .non-word {{
                        background-color: #ffcccc;
                        border-bottom: 2px solid red;
                        padding: 2px;
                        border-radius: 3px;
                        color: #000;
                        cursor: pointer;
                    }}
                    .real-word {{
                        background-color: #ffe6cc;
                        border-bottom: 2px solid #ff9900;
                        padding: 2px;
                        border-radius: 3px;
                        color: #000;
                        cursor: pointer;
                    }}
                    </style>
                    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; margin-bottom: 20px; line-height: 1.8; font-size: 16px;">
                    {html_text}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add JavaScript for click handling
                    st.markdown("""
                    <script>
                    function handleWordClick(wordIndex) {
                        // This would typically send a message to Streamlit
                        // For now, we'll just alert the word index
                        alert("Clicked on word index: " + wordIndex);
                    }
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Legend for error types
                    st.caption("Red underline: Spelling errors | Orange underline: Confusable words | Click on any highlighted word for suggestions")
                    
                    # Show corrections for each error
                    if errors:
                        st.subheader("Suggested Corrections")
                        
                        # Create subtabs for different error types
                        subtab1, subtab2 = st.tabs(["Spelling Errors", "Confusable Words"])
                        
                        with subtab1:
                            spelling_errors = [err for err in errors if err[1] == "non-word"]
                            if spelling_errors:
                                for i, (word, start, end, error_type) in [(idx, error_info[idx]) for idx, _ in spelling_errors]:
                                    # Get context
                                    prev_word = words[i-1].lower() if i > 0 and words[i-1].isalpha() else None
                                    next_word = words[i+1].lower() if i < len(words)-1 and words[i+1].isalpha() else None
                                    
                                    # Get suggestions with detailed stats
                                    suggestions = corrector.suggest_corrections_with_stats(word.lower(), prev_word, next_word)
                                    
                                    if suggestions:
                                        st.write(f"**{word}** → Suggestions with detailed analysis:")
                                        
                                        # Create a table for the suggestions
                                        suggestion_data = []
                                        for idx, suggestion in enumerate(suggestions):
                                            suggestion_data.append({
                                                "Rank": idx + 1,
                                                "Suggestion": suggestion['candidate'],
                                                "Frequency": suggestion['frequency'],
                                                "Edit Distance": suggestion['edit_distance'],
                                                "Context Score": f"{suggestion['score']:.2f}",
                                                "P(prev|word)": f"{suggestion['prev_prob']:.4f}" if suggestion['prev_prob'] > 0 else "N/A",
                                                "P(word|next)": f"{suggestion['next_prob']:.4f}" if suggestion['next_prob'] > 0 else "N/A"
                                            })
                                        
                                        # Display the table
                                        st.table(suggestion_data)
                                        
                                        # Explain why the top suggestion is best
                                        if len(suggestions) > 0:
                                            best = suggestions[0]
                                            st.write(f"**Why '{best['candidate']}' is the best suggestion:**")
                                            
                                            reasons = []
                                            if best['frequency'] > 0:
                                                reasons.append(f"High frequency ({best['frequency']} occurrences in corpus)")
                                            if best['edit_distance'] == 1:
                                                reasons.append("Only 1 edit away from the original")
                                            elif best['edit_distance'] == 2:
                                                reasons.append("Only 2 edits away from the original")
                                            if best['prev_prob'] > 0.01:
                                                reasons.append(f"Fits well with previous word (P={best['prev_prob']:.4f})")
                                            if best['next_prob'] > 0.01:
                                                reasons.append(f"Fits well with next word (P={best['next_prob']:.4f})")
                                            
                                            if reasons:
                                                for reason in reasons:
                                                    st.write(f"- {reason}")
                                            else:
                                                st.write("It has the highest overall score considering all factors")
                                    else:
                                        st.write(f"**{word}** → No suggestions found")
                            else:
                                st.success("No spelling errors found!")
                        
                        with subtab2:
                            real_word_errors = [err for err in errors if err[1] == "real-word"]
                            if real_word_errors:
                                for i, (word, start, end, error_type) in [(idx, error_info[idx]) for idx, _ in real_word_errors]:
                                    # Get context
                                    prev_word = words[i-1].lower() if i > 0 and words[i-1].isalpha() else None
                                    next_word = words[i+1].lower() if i < len(words)-1 and words[i+1].isalpha() else None
                                    
                                    # Get suggestions with detailed stats
                                    suggestions = corrector.suggest_corrections_with_stats(word.lower(), prev_word, next_word)
                                    
                                    if suggestions:
                                        st.write(f"**{word}** → Common confusions and alternatives:")
                                        
                                        # Create a table for the suggestions
                                        suggestion_data = []
                                        for idx, suggestion in enumerate(suggestions):
                                            suggestion_data.append({
                                                "Rank": idx + 1,
                                                "Suggestion": suggestion['candidate'],
                                                "Frequency": suggestion['frequency'],
                                                "Confusion Weight": f"{suggestion['confusable_weight']:.2f}" if suggestion['confusable_weight'] > 0 else "N/A",
                                                "Context Score": f"{suggestion['score']:.2f}",
                                                "P(prev|word)": f"{suggestion['prev_prob']:.4f}" if suggestion['prev_prob'] > 0 else "N/A",
                                                "P(word|next)": f"{suggestion['next_prob']:.4f}" if suggestion['next_prob'] > 0 else "N/A"
                                            })
                                        
                                        # Display the table
                                        st.table(suggestion_data)
                                    else:
                                        st.write(f"**{word}** → No suggestions found")
                            else:
                                st.success("No confusable words found!")
                    
                    else:
                        st.success("No errors detected! Your text looks great!")
                        
                        # Show some stats about the text
                        word_count = len([w for w in words if w.isalpha()])
                        unique_words = len(set([w.lower() for w in words if w.isalpha()]))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Word Count", word_count)
                        with col2:
                            st.metric("Unique Words", unique_words)
            
            else:
                st.warning("Please enter some text to check.")
    
    with tab2:
        st.header("Word Explorer")
        st.markdown("Explore all words in the corpus vocabulary.")
        
        # Search functionality
        search_query = st.text_input("Search for words:", placeholder="Enter a word or part of a word")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort by:", ["Frequency (High to Low)", "Alphabetical (A-Z)", "Alphabetical (Z-A)"])
        with col2:
            words_per_page = st.slider("Words per page:", 10, 100, 20)
        
        # Get the word list to display
        if search_query:
            word_list = corrector.search_vocabulary(search_query)
            st.write(f"Found {len(word_list)} words matching '{search_query}'")
        else:
            word_list = st.session_state.vocab_list
            st.write(f"Total words in vocabulary: {len(word_list):,}")
        
        # Sort the word list
        if sort_by == "Alphabetical (A-Z)":
            word_list.sort(key=lambda x: x[0].lower())
        elif sort_by == "Alphabetical (Z-A)":
            word_list.sort(key=lambda x: x[0].lower(), reverse=True)
        
        # Pagination
        if word_list:
            total_pages = (len(word_list) // words_per_page) + (1 if len(word_list) % words_per_page > 0 else 0)
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * words_per_page
            end_idx = min(start_idx + words_per_page, len(word_list))
            
            # Display words in a table
            word_data = []
            for word, freq in word_list[start_idx:end_idx]:
                word_data.append({
                    "Word": word,
                    "Frequency": freq,
                    "Probability": f"{corrector.P(word):.6f}"
                })
            
            st.table(word_data)
            
            # Page navigation
            if total_pages > 1:
                st.write(f"Page {page} of {total_pages}")
        else:
            st.info("No words found. Try a different search query.")
    
    with tab3:
        st.header("ℹAbout this System")
        st.markdown("""
        This spelling correction system uses:
        
        - **N-gram language model** trained on scientific literature arVix Corpus
        - **Edit distance algorithm** for spelling suggestions
        - **Confusable words database** for common mistakes
        
        The system can detect:
        - **Spelling errors**: Words not found in the vocabulary
        - **Confusable words**: Correctly spelled words that are often confused with others
        
        **How suggestions are ranked:**
        - **Frequency**: How often the word appears in the corpus
        - **Edit Distance**: How many changes needed from the original word
        - **Context**: How well the word fits with surrounding words
        - **Confusion Weight**: For commonly confused words, how likely the substitution is
        """)
        
        if 'corrector' in st.session_state:
            st.metric("Vocabulary Size", f"{len(st.session_state.corrector.vocab):,}")
            st.metric("Total Words in Corpus", f"{st.session_state.corrector.total_words:,}")

if __name__ == "__main__":
    main()
