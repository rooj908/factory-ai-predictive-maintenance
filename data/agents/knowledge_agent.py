from rag.retriever import retrieve_documents


class KnowledgeAgent:

    name = "Knowledge Agent"

    def analyze(self, query):

        results = retrieve_documents(
            query,
            top_k=2
        )

        evidence = []

        for result in results:
            evidence.append({
                "source": result["title"],
                "score": round(result["score"], 3),
                "content": result["content"].strip()
            })

        return {
            "agent": self.name,
            "evidence": evidence
        }
