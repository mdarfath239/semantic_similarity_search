
from src.inference import semantic_search

# ==========================================
# 1. TEST QUERIES
# ==========================================

test_queries = [
    "computer graphics and image processing",
    "space exploration and NASA",
    "motorcycles and automobiles",
    "medical science and health",
    "computer programming and software"
]

# ==========================================
# 2. EVALUATE SEARCH RESULTS
# ==========================================

for query in test_queries:

    print("\n" + "=" * 70)
    print("SEARCH QUERY:", query)
    print("=" * 70)

    results = semantic_search(
        query=query,
        top_n=5
    )

    if not results:
        print("No matching documents found.")
        continue

    # ======================================
    # 3. DISPLAY TOP 5 RESULTS
    # ======================================

    for result in results:

        print("\nRank:", result["rank"])

        print("Category:", result["category"])

        print(
            "Similarity Score:",
            round(result["similarity_score"], 4)
        )

        print(
            "Document:",
            str(result["text"])[:300]
        )

# ==========================================
# 4. EVALUATION COMPLETED
# ==========================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)
