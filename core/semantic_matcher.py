"""
Semantic Matching using Sentence Transformers
Provides semantic similarity scoring for text deduplication and entity resolution
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Lazy import sentence transformers to avoid slow startup
_model_cache = {}

# Embedding cache: keyed by (model_name + md5 of texts), stores np.ndarray
# Persists for the lifetime of the Streamlit server process.
_embedding_cache: Dict[str, np.ndarray] = {}
_EMBEDDING_CACHE_MAX = 20  # max cached arrays (each ~few MB for 500 rows)


def _embedding_cache_key(texts: List[str], model_name: str) -> str:
    digest = hashlib.md5("|".join(texts).encode("utf-8", errors="replace")).hexdigest()
    return f"{model_name}:{digest}"


def get_sentence_transformer_model(model_name: str = 'all-MiniLM-L6-v2'):
    """
    Load sentence transformer model with caching

    Args:
        model_name: Model name from sentence-transformers
                   - 'all-MiniLM-L6-v2': Fast, small (80MB), good quality
                   - 'all-mpnet-base-v2': Slower, larger (420MB), best quality

    Returns:
        SentenceTransformer model
    """
    global _model_cache

    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model_cache[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise ImportError(
                f"Failed to load sentence transformer model '{model_name}'. "
                f"Error: {str(e)}. "
                f"Make sure sentence-transformers is installed: pip install sentence-transformers"
            )

    return _model_cache[model_name]


def encode_texts(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    show_progress: bool = False
) -> np.ndarray:
    """
    Encode list of texts into embeddings

    Args:
        texts: List of text strings to encode
        model_name: Sentence transformer model to use
        batch_size: Batch size for encoding
        show_progress: Show progress bar

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    global _embedding_cache

    # Clean texts first so cache key reflects actual input to the model
    texts_clean = [str(t).strip() if pd.notna(t) else "" for t in texts]

    cache_key = _embedding_cache_key(texts_clean, model_name)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    model = get_sentence_transformer_model(model_name)

    # Encode
    embeddings = model.encode(
        texts_clean,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )

    # Evict oldest entry if cache is full (insertion-order dict, Python 3.7+)
    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
        oldest = next(iter(_embedding_cache))
        del _embedding_cache[oldest]
    _embedding_cache[cache_key] = embeddings

    return embeddings


def compute_semantic_similarity(
    texts_a: List[str],
    texts_b: Optional[List[str]] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute semantic similarity between two sets of texts

    Args:
        texts_a: First set of texts
        texts_b: Second set of texts (if None, compute pairwise similarity within texts_a)
        model_name: Sentence transformer model to use
        batch_size: Batch size for encoding

    Returns:
        Similarity matrix of shape (len(texts_a), len(texts_b))
    """
    # Encode texts
    embeddings_a = encode_texts(texts_a, model_name, batch_size)

    if texts_b is None:
        # Pairwise similarity within texts_a
        similarity_matrix = cosine_similarity(embeddings_a)
    else:
        # Cross similarity between texts_a and texts_b
        embeddings_b = encode_texts(texts_b, model_name, batch_size)
        similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)

    return similarity_matrix


def hybrid_similarity_tfidf_transformer(
    df: pd.DataFrame,
    columns: List[str],
    vec_pack: Dict[str, Any],
    threshold_high: float = 0.9,
    threshold_low: float = 0.6,
    model_name: str = 'all-MiniLM-L6-v2',
    use_transformer_for_medium: bool = True
) -> pd.DataFrame:
    """
    Hybrid approach: TF-IDF for initial filtering, Sentence Transformer for refinement

    Strategy:
    1. Compute TF-IDF similarity for all pairs (fast)
    2. High confidence (>0.9): Accept as duplicates
    3. Low confidence (<0.6): Reject as non-duplicates
    4. Medium confidence (0.6-0.9): Re-score with Sentence Transformer

    Args:
        df: Input dataframe
        columns: Columns to compare
        vec_pack: TF-IDF vectorizer pack from build_vectorizers
        threshold_high: High confidence threshold (auto-accept)
        threshold_low: Low confidence threshold (auto-reject)
        model_name: Sentence transformer model name
        use_transformer_for_medium: Use transformer for medium confidence pairs

    Returns:
        DataFrame with columns: idxA, idxB, score_tfidf, score_semantic, score_final
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Step 1: Compute TF-IDF similarity (fast)
    # vec_pack contains pre-computed vectors, not vectorizers
    # Use the row_concat_vec directly
    row_vectors = vec_pack.get("row_concat_vec")

    if row_vectors is None or row_vectors.shape[1] == 0:
        # Fallback: compute TF-IDF on the fly
        from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
        texts = df[columns].astype(str).agg(' '.join, axis=1).tolist()
        vec = TfIdf(analyzer='char', ngram_range=(3, 5))
        row_vectors = vec.fit_transform(texts).toarray()

    tfidf_similarity = cosine_similarity(row_vectors)

    # Extract pairs
    n = len(df)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score_tfidf = tfidf_similarity[i, j]
            pairs.append({
                'idxA': i,
                'idxB': j,
                'score_tfidf': score_tfidf,
                'confidence_level': 'unknown'
            })

    pairs_df = pd.DataFrame(pairs)

    if len(pairs_df) == 0:
        return pairs_df

    # Step 2: Categorize by confidence
    high_conf = pairs_df['score_tfidf'] >= threshold_high
    low_conf = pairs_df['score_tfidf'] < threshold_low
    medium_conf = ~high_conf & ~low_conf

    pairs_df.loc[high_conf, 'confidence_level'] = 'high'
    pairs_df.loc[low_conf, 'confidence_level'] = 'low'
    pairs_df.loc[medium_conf, 'confidence_level'] = 'medium'

    # Initialize final scores with TF-IDF scores
    pairs_df['score_semantic'] = np.nan
    pairs_df['score_final'] = pairs_df['score_tfidf']

    # Step 3: Re-score medium confidence pairs with Sentence Transformer
    if use_transformer_for_medium and medium_conf.sum() > 0:
        medium_pairs = pairs_df[medium_conf].copy()

        # Get unique indices for semantic encoding
        unique_indices = pd.unique(
            medium_pairs[['idxA', 'idxB']].values.ravel()
        )

        # Concatenate text from selected columns
        texts = df.iloc[unique_indices][columns].astype(str).agg(' '.join, axis=1).tolist()

        # Encode with sentence transformer
        embeddings = encode_texts(texts, model_name)

        # Create index mapping
        idx_to_embedding_pos = {idx: pos for pos, idx in enumerate(unique_indices)}

        # Compute semantic similarity for medium pairs
        semantic_scores = []
        for _, row in medium_pairs.iterrows():
            pos_a = idx_to_embedding_pos[row['idxA']]
            pos_b = idx_to_embedding_pos[row['idxB']]

            emb_a = embeddings[pos_a].reshape(1, -1)
            emb_b = embeddings[pos_b].reshape(1, -1)

            sem_score = cosine_similarity(emb_a, emb_b)[0, 0]
            semantic_scores.append(sem_score)

        # Update scores for medium confidence pairs
        pairs_df.loc[medium_conf, 'score_semantic'] = semantic_scores
        pairs_df.loc[medium_conf, 'score_final'] = semantic_scores

    return pairs_df


def semantic_deduplicate(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.85,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Find duplicate pairs using semantic similarity

    Args:
        df: Input dataframe
        columns: Columns to compare
        threshold: Similarity threshold for considering duplicates
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding

    Returns:
        DataFrame with duplicate pairs (idxA, idxB, score)
    """
    # Concatenate selected columns
    texts = df[columns].astype(str).agg(' '.join, axis=1).tolist()

    # Encode
    embeddings = encode_texts(texts, model_name, batch_size, show_progress=True)

    # Compute similarity
    similarity_matrix = cosine_similarity(embeddings)

    # Extract pairs above threshold
    n = len(df)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score = similarity_matrix[i, j]
            if score >= threshold:
                pairs.append({
                    'idxA': i,
                    'idxB': j,
                    'score': score
                })

    return pd.DataFrame(pairs)


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available sentence transformer models with metadata

    Returns:
        Dictionary of model name -> metadata
    """
    return {
        'all-MiniLM-L6-v2': {
            'name': 'all-MiniLM-L6-v2',
            'size_mb': 80,
            'speed': 'Fast',
            'quality': 'Good',
            'description': 'Small, fast model. Good for most use cases.',
            'recommended': True
        },
        'all-mpnet-base-v2': {
            'name': 'all-mpnet-base-v2',
            'size_mb': 420,
            'speed': 'Slower',
            'quality': 'Best',
            'description': 'Larger, slower model. Best quality for critical data.',
            'recommended': False
        },
        'paraphrase-multilingual-MiniLM-L12-v2': {
            'name': 'paraphrase-multilingual-MiniLM-L12-v2',
            'size_mb': 420,
            'speed': 'Slower',
            'quality': 'Good',
            'description': 'Supports 50+ languages.',
            'recommended': False
        }
    }


def check_model_availability(model_name: str) -> bool:
    """Check if a model is available/downloadable"""
    try:
        get_sentence_transformer_model(model_name)
        return True
    except Exception:
        return False
