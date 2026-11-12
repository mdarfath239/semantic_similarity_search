
from src.inference import semantic_search


# ==========================================
# 1. TEST SEARCH QUERY
# ==========================================

query = "computer graphics and image processing"


# ==========================================
# 2. PERFORM SEMANTIC SEARCH
# ==========================================

results = semantic_search(
    query=query,
    top_n=5
)


# ==========================================
# 3. DISPLAY SEARCH RESULTS
# ==========================================

print("\n" + "=" * 70)

print("SEMANTIC SIMILARITY SEARCH RESULTS")

print("=" * 70)

print("\nSearch Query:", query)

print("\nTotal Results:", len(results))


# ==========================================
# 4. DISPLAY TOP MATCHING DOCUMENTS
# ==========================================

for result in results:

    print("\n" + "-" * 70)

    print("Rank:", result["rank"])

    print("Document ID:", result["document_id"])

    print("Category:", result["category"])

    print(
        "Similarity Score:",
        round(result["similarity_score"], 4)
    )

    print("\nDocument Text:")

    print(str(result["text"])[:500])


# ==========================================
# 5. TEST COMPLETED
# ==========================================

print("\n" + "=" * 70)

print("SEMANTIC SEARCH TEST COMPLETED")

print("=" * 70)
