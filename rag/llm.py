import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_maintenance_explanation(
    failure_probability,
    retrieved_results
):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env file"
        )

    client = Groq(
        api_key=api_key
    )

    context = "\n\n".join(
        [
            f"Source: {result['title']}\n"
            f"{result['content']}"
            for result in retrieved_results
        ]
    )

    prompt = f"""
You are an industrial predictive maintenance assistant.

The predictive maintenance model produced:

Failure probability: {failure_probability:.2%}

Retrieved maintenance evidence:

{context}

Using ONLY the retrieved evidence, provide:

1. Risk Summary
2. Recommended Action
3. Safety Precaution
4. Sources Used

Do not invent procedures or facts.

The recommendation is decision support only.
A human supervisor must approve the final operational decision.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content":
                "You are an evidence-based industrial "
                "maintenance assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    from retriever import retrieve_documents

    failure_probability = 0.78

    query = """
    Machine has high vibration and elevated failure risk.
    What maintenance action should be taken?
    """

    results = retrieve_documents(
        query,
        top_k=2
    )

    answer = generate_maintenance_explanation(
        failure_probability,
        results
    )

    print("\n" + "=" * 60)
    print("RAG + LLM MAINTENANCE EXPLANATION")
    print("=" * 60)

    print(answer)
