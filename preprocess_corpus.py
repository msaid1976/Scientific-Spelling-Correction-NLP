import json
import re
from collections import Counter, defaultdict
import os
import time

def tokenize(text):
    """Tokenize text into words"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def _emit_progress(progress_callback, stage, progress, message, details=None):
    """Emit structured progress updates when a callback is provided."""
    payload = {
        "stage": stage,
        "progress": max(0.0, min(1.0, float(progress))),
        "message": message,
        "details": details or {},
    }
    if progress_callback:
        progress_callback(payload)


def preprocess_corpus(
    corpus_path,
    output_dir="preprocessed_data",
    min_words=100000,
    min_word_frequency=1,
    progress_callback=None,
):
    """Preprocess the corpus and save n-gram data with optional vocabulary pruning."""
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    unigrams = Counter()
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(lambda: defaultdict(Counter))
    total_words = 0

    corpus_size_bytes = os.path.getsize(corpus_path) if os.path.exists(corpus_path) else 0
    _emit_progress(
        progress_callback,
        "setup",
        0.20,
        "Starting corpus preprocessing.",
        {
            "corpus_path": corpus_path,
            "target_words": int(min_words),
            "min_word_frequency": int(min_word_frequency),
            "corpus_size_mb": round(corpus_size_bytes / (1024 * 1024), 2),
        },
    )
    
    # Process the corpus
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        line_idx = -1
        for line_idx, line in enumerate(f):
            try:
                data = json.loads(line)
                text = data.get('title', '') + ' ' + data.get('abstract', '')
                words = tokenize(text)
                
                # Update counts
                word_count = len(words)
                total_words += word_count
                
                # Update unigrams
                for i, word in enumerate(words):
                    unigrams[word] += 1
                    # Update bigrams
                    if i < len(words)-1:
                        bigrams[words[i]][words[i+1]] += 1
                    # Update trigrams
                    if i < len(words)-2:
                        trigrams[words[i]][words[i+1]][words[i+2]] += 1
            except json.JSONDecodeError:
                continue
            
            # Emit progress every 5,000 lines
            if line_idx % 5000 == 0:
                progress_ratio = min(1.0, total_words / max(1, min_words))
                _emit_progress(
                    progress_callback,
                    "processing",
                    0.20 + (0.45 * progress_ratio),
                    "Tokenizing and counting n-grams.",
                    {
                        "lines_processed": int(line_idx),
                        "words_processed": int(total_words),
                        "current_vocab_size": int(len(unigrams)),
                    },
                )
            
            # Stop if we've reached our target
            if total_words >= min_words:
                _emit_progress(
                    progress_callback,
                    "processing",
                    0.66,
                    "Reached target corpus size.",
                    {
                        "lines_processed": int(line_idx),
                        "words_processed": int(total_words),
                    },
                )
                break
    
    raw_total_words = int(total_words)
    raw_vocab_size = int(len(unigrams))

    # Apply minimum frequency threshold to vocabulary and n-grams
    _emit_progress(
        progress_callback,
        "pruning",
        0.70,
        "Applying minimum word-frequency threshold.",
        {
            "min_word_frequency": int(min_word_frequency),
            "raw_vocab_size": raw_vocab_size,
        },
    )

    filtered_vocab = {
        word for word, count in unigrams.items() if count >= max(1, int(min_word_frequency))
    }
    unigrams = Counter(
        {word: count for word, count in unigrams.items() if word in filtered_vocab}
    )
    filtered_total_words = int(sum(unigrams.values()))

    filtered_bigrams = defaultdict(Counter)
    bigram_heads = list(bigrams.items())
    total_bigram_heads = max(1, len(bigram_heads))
    for idx, (w1, next_words) in enumerate(bigram_heads, start=1):
        if w1 not in filtered_vocab:
            continue
        kept_next = {w2: c for w2, c in next_words.items() if w2 in filtered_vocab}
        if kept_next:
            filtered_bigrams[w1] = Counter(kept_next)
        if idx % 5000 == 0 or idx == total_bigram_heads:
            _emit_progress(
                progress_callback,
                "pruning",
                0.72 + (0.06 * (idx / total_bigram_heads)),
                "Filtering bigrams by retained vocabulary.",
                {
                    "bigram_heads_processed": int(idx),
                    "bigram_heads_total": int(total_bigram_heads),
                },
            )
    bigrams = filtered_bigrams

    filtered_trigrams = defaultdict(lambda: defaultdict(Counter))
    trigram_heads = list(trigrams.items())
    total_trigram_heads = max(1, len(trigram_heads))
    for idx, (w1, nested) in enumerate(trigram_heads, start=1):
        if w1 not in filtered_vocab:
            continue
        for w2, subnested in nested.items():
            if w2 not in filtered_vocab:
                continue
            kept_next = {w3: c for w3, c in subnested.items() if w3 in filtered_vocab}
            if kept_next:
                filtered_trigrams[w1][w2] = Counter(kept_next)
        if idx % 5000 == 0 or idx == total_trigram_heads:
            _emit_progress(
                progress_callback,
                "pruning",
                0.78 + (0.10 * (idx / total_trigram_heads)),
                "Filtering trigrams by retained vocabulary.",
                {
                    "trigram_heads_processed": int(idx),
                    "trigram_heads_total": int(total_trigram_heads),
                },
            )
    trigrams = filtered_trigrams

    vocab = sorted(filtered_vocab)
    _emit_progress(
        progress_callback,
        "pruning",
        0.88,
        "Vocabulary pruning complete.",
        {
            "raw_words": raw_total_words,
            "retained_words": filtered_total_words,
            "raw_vocab_size": raw_vocab_size,
            "retained_vocab_size": int(len(vocab)),
        },
    )
    
    # Save preprocessed data
    _emit_progress(progress_callback, "saving", 0.90, "Saving preprocessed artifacts.")
    
    with open(os.path.join(output_dir, "unigrams.json"), 'w', encoding='utf-8') as f:
        json.dump(dict(unigrams), f, ensure_ascii=False)
    _emit_progress(progress_callback, "saving", 0.92, "Saved unigrams.json.")
    
    with open(os.path.join(output_dir, "bigrams.json"), 'w', encoding='utf-8') as f:
        # Convert defaultdict to regular dict for JSON serialization
        bigrams_dict = {k: dict(v) for k, v in bigrams.items()}
        json.dump(bigrams_dict, f, ensure_ascii=False)
    _emit_progress(progress_callback, "saving", 0.94, "Saved bigrams.json.")
    
    with open(os.path.join(output_dir, "trigrams.json"), 'w', encoding='utf-8') as f:
        # Convert nested defaultdicts to regular dicts for JSON serialization
        trigrams_dict = {}
        for w1, nested in trigrams.items():
            trigrams_dict[w1] = {}
            for w2, subnested in nested.items():
                trigrams_dict[w1][w2] = dict(subnested)
        json.dump(trigrams_dict, f, ensure_ascii=False)
    _emit_progress(progress_callback, "saving", 0.96, "Saved trigrams.json.")
    
    with open(os.path.join(output_dir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)
    _emit_progress(progress_callback, "saving", 0.97, "Saved vocab.json.")
    
    metadata = {
        "total_words": filtered_total_words,
        "raw_total_words": raw_total_words,
        "vocab_size": len(vocab),
        "raw_vocab_size": raw_vocab_size,
        "lines_processed": int(line_idx + 1),
        "target_words": int(min_words),
        "min_word_frequency": int(min_word_frequency),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
    _emit_progress(progress_callback, "saving", 0.99, "Saved metadata.json.", metadata)
    
    _emit_progress(
        progress_callback,
        "complete",
        1.00,
        "Preprocessing complete.",
        metadata,
    )
    return metadata

if __name__ == "__main__":
    corpus_path = "corpus/arxiv-metadata-oai-snapshot.json"  # Update this path
    preprocess_corpus(corpus_path, min_words=30000000, min_word_frequency=1)
    

