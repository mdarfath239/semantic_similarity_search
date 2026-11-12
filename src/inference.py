
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "word2vec_model.model"

VECTORS_PATH = ROOT / "models" / "search_document_vectors.pkl"

METADATA_PATH = ROOT / "models" / "search_document_metadata.pkl"


# ==========================================
# 2. LOAD TRAINED WORD2VEC MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Word2Vec model not found: {MODEL_PATH}"
        )

    model = Word2Vec.load(str(MODEL_PATH))

    print("Word2Vec model loaded successfully!")
    print("Vector size:", model.vector_size)

    return model


# ==========================================
# 3. LOAD SEARCH DOCUMENT VECTORS
# ==========================================

def load_document_vectors():

    if not VECTORS_PATH.exists():
        raise FileNotFoundError(
            f"Document vectors not found: {VECTORS_PATH}"
        )

    vectors = joblib.load(VECTORS_PATH)

    if isinstance(vectors, pd.DataFrame):
        vectors = vectors.to_numpy()

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D document vectors, got {vectors.shape}"
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Document vectors contain invalid numeric values."
        )

    print("Document vectors loaded successfully!")
    print("Document vector shape:", vectors.shape)

    return vectors


# ==========================================
# 4. LOAD SEARCH DOCUMENT METADATA
# ==========================================

def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Document metadata not found: {METADATA_PATH}"
        )

    metadata = joblib.load(METADATA_PATH)

    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame(metadata)

    metadata = metadata.reset_index(drop=True)

    required_columns = [
        "document_id",
        "category",
        "document_text"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    print("Document metadata loaded successfully!")
    print("Metadata shape:", metadata.shape)
    print("Metadata columns:", metadata.columns.tolist())

    return metadata


# ==========================================
# 5. LOAD ALL MODEL FILES
# ==========================================

MODEL = load_model()

DOCUMENT_VECTORS = load_document_vectors()

DOCUMENT_METADATA = load_metadata()


# ==========================================
# 6. VALIDATE MODEL AND DOCUMENT DATA
# ==========================================

if len(DOCUMENT_VECTORS) != len(DOCUMENT_METADATA):

    raise ValueError(
        "Document vectors and metadata have different row counts."
    )


if DOCUMENT_VECTORS.shape[1] != MODEL.vector_size:

    raise ValueError(
        "Document vector dimensions do not match Word2Vec model."
    )


print("\nAll model files loaded successfully!")

print("Total search documents:", len(DOCUMENT_VECTORS))

print("Word2Vec dimensions:", MODEL.vector_size)


# ==========================================
# 7. CONVERT QUERY INTO WORD2VEC VECTOR
# ==========================================

def get_query_vector(query):

    tokens = preprocess_text(query)

    if not tokens:
        return None

    word_vectors = [

        MODEL.wv[word]

        for word in tokens

        if word in MODEL.wv

    ]

    if not word_vectors:
        return None

    query_vector = np.mean(
        word_vectors,
        axis=0
    )

    return query_vector.astype(np.float32)


# ==========================================
# 8. SEMANTIC SIMILARITY SEARCH
# ==========================================

def semantic_search(query, top_n=5):

    if not isinstance(query, str) or not query.strip():

        raise ValueError(
            "Please enter a valid search query."
        )

    if not isinstance(top_n, int) or top_n < 1:

        raise ValueError(
            "top_n must be a positive integer."
        )

    # Convert query into Word2Vec vector

    query_vector = get_query_vector(query)

    if query_vector is None:

        return []

    # Calculate cosine similarity

    similarities = cosine_similarity(

        query_vector.reshape(1, -1),

        DOCUMENT_VECTORS

    )[0]

    # Get top matching document indexes

    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = []

    # Retrieve matching documents

    for rank, idx in enumerate(top_indices, start=1):

        document = DOCUMENT_METADATA.iloc[int(idx)]

        results.append({

            "rank": rank,

            "document_id": document["document_id"],

            "category": document["category"],

            "text": document["document_text"],

            "similarity_score": float(similarities[idx])

        })

    return results


# ==========================================
# 9. INTERACTIVE TERMINAL SEARCH
# ==========================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)

    print("SEMANTIC SIMILARITY SEARCH SYSTEM")

    print("=" * 60)

    print("\nModel is ready!")

    print("Dataset: 20 Newsgroups")

    print("Total documents:", len(DOCUMENT_VECTORS))

    print("\nType 'exit' to stop the application.")

    while True:

        query = input(
            "\nEnter your search query: "
        ).strip()

        if query.lower() in ["exit", "quit"]:

            print("\nSearch application closed.")

            break

        try:

            results = semantic_search(
                query,
                top_n=5
            )

            if not results:

                print(
                    "\nNo matching documents found."
                )

                print(
                    "Try another query."
                )

                continue

            print("\n")
            print("=" * 60)

            print("SEARCH RESULTS")

            print("=" * 60)

            for result in results:

                print("\n")
                print("-" * 60)

                print(
                    "Rank:",
                    result["rank"]
                )

                print(
                    "Document ID:",
                    result["document_id"]
                )

                print(
                    "Category:",
                    result["category"]
                )

                print(
                    "Similarity Score:",
                    round(
                        result["similarity_score"],
                        4
                    )
                )

                print("\nDocument Text:")

                print(
                    str(result["text"])[:500]
                )

        except Exception as error:

            print(
                "\nSearch error:",
                error
            )
