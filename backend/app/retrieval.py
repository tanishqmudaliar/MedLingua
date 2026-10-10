import math
import re
from collections import Counter

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours",
}


def _tokenize(text: str) -> list[str]:
    return [
        word.casefold()
        for word in re.findall(r"\b[a-zA-Z0-9'-]+\b", text)
        if word.casefold() not in _STOP_WORDS and len(word) > 1
    ]


def chunk_text(text: str, max_words_per_chunk: int = 250, overlap_words: int = 40) -> list[str]:
    """Divides document text into semantic paragraph or window chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_words: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(current_words) + len(words) <= max_words_per_chunk:
            current_words.extend(words)
        else:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap_words:] + words
            else:
                chunks.append(" ".join(words[:max_words_per_chunk]))
                current_words = words[max_words_per_chunk:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks if chunks else [text]


def retrieve_relevant_chunks(
    document_text: str, query: str, top_k: int = 3, max_total_words: int = 1000
) -> list[str]:
    """Retrieves top-k relevant document chunks using memory-efficient BM25-style scoring."""
    words = document_text.split()
    if len(words) <= max_total_words:
        return [document_text.strip()] if document_text.strip() else []

    chunks = chunk_text(document_text, max_words_per_chunk=250, overlap_words=40)
    if len(chunks) <= top_k:
        return chunks

    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    chunk_token_lists = [_tokenize(chunk) for chunk in chunks]
    total_docs = len(chunks)
    doc_freqs: Counter[str] = Counter()
    for tokens in chunk_token_lists:
        doc_freqs.update(set(tokens))

    avg_dl = sum(len(tokens) for tokens in chunk_token_lists) / max(1, total_docs)
    k1 = 1.5
    b = 0.75

    scores: list[tuple[float, int]] = []
    for idx, tokens in enumerate(chunk_token_lists):
        if not tokens:
            scores.append((0.0, idx))
            continue
        term_counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for q in query_tokens:
            if q in term_counts:
                tf = term_counts[q]
                df = doc_freqs.get(q, 0)
                idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_dl)))
                score += idf * (numerator / denominator)
        scores.append((score, idx))

    scores.sort(key=lambda item: item[0], reverse=True)
    selected_indices = sorted([idx for _, idx in scores[:top_k]])
    return [chunks[i] for i in selected_indices]
