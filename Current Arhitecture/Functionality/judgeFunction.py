# SCORING SYSTEM LLM
import requests


def judge_scores(question, context_chunks, answer):
    context = "\n\n--- FRAGMENT NOU ---\n\n".join(context_chunks)

    prompt = f"""
You are a ruthless and objective evaluator of a RAG (Retrieval-Augmented Generation) chatbot for university admissions.
Your task is to evaluate the chatbot's answer based STRICTLY on the provided Reference Context.

Question:
{question}

Chatbot Answer:
{answer}

Reference Context:
{context}

---
SCORING RUBRIC (Be extremely strict!):
1. Factual correctness (0-10): 
   - 10 = Perfectly matches the context.
   - 5 = Partially correct, but contains errors.
   - 0 = The information contradicts the context or is completely invented.
2. Completeness (0-10): Did it fully answer the user's prompt based on available context?
3. Clarity (0-10): Is the answer easy to read and in Romanian? (Score 0 if the answer is in English!).
4. Hallucinations penalty (0-10): 
   - 10 = NO hallucinations at all. Only used the context.
   - 0 = The model invented rules, formulas, or deadlines NOT present in the Reference Context.

If the Chatbot Answer is exactly "Nu dețin această informație conform datelor extrase de pe site-ul oficial." AND the answer is indeed NOT present in the Reference Context, you MUST score Factual=10, Complete=10, and Hallucination=10. Do not penalize honesty!

Return ONLY valid JSON format:
{{
  "factual": <int>,
  "complete": <int>,
  "clarity": <int>,
  "no_hallucination": <int>,
  "brief_reason": "<1-2 sentences explaining exactly why points were deducted>"
}}
"""

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": "google/gemma-2-9b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )


    data = response.json()
    
    if "choices" not in data:
        print("Eroare primită de la LM Studio:", data)
        return "Eroare la generarea răspunsului."

    return data["choices"][0]["message"]["content"]