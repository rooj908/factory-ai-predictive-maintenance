from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    {
        "title": "High Vibration Maintenance Guide",
        "content": """
High vibration may indicate bearing wear, shaft misalignment,
imbalance or mechanical looseness.

Inspect bearings, coupling and shaft alignment.

If vibration continues after inspection, schedule maintenance
before returning the machine to full production.
"""
    },
    {
        "title": "Preventive Maintenance SOP",
        "content": """
Machines should be inspected regularly based on operating hours,
sensor trends and maintenance history.

Record maintenance actions and monitor vibration, temperature,
pressure and current after servicing.
"""
    },
    {
        "title": "Electrical Safety SOP",
        "content": """
Before inspecting electrical components, isolate the machine
from the power source and follow lockout/tagout procedures.

Inspect cables, connections and motor current abnormalities.

Only authorized personnel should perform electrical maintenance.
"""
    }
]

texts = [doc["content"] for doc in documents]

vectorizer = TfidfVectorizer(
    stop_words="english"
)

doc_vectors = vectorizer.fit_transform(texts)


def retrieve_documents(query, top_k=2):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        doc_vectors
    )[0]

    indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in indices:

        results.append({
            "title": documents[index]["title"],
            "content": documents[index]["content"],
            "score": float(scores[index])
        })

    return results


if __name__ == "__main__":

    query = "machine has high vibration and needs maintenance"

    results = retrieve_documents(query)

    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    for result in results:

        print("\nSOURCE:", result["title"])
        print("SCORE:", round(result["score"], 3))
        print(result["content"])
