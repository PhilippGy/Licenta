# LOAD
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import numpy as np
import requests
import json


chunks = np.load("../Data/chunks.npy", allow_pickle=True)
index = faiss.read_index("../Data/index.faiss")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
reranker_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")



# RETRIEVAL 
# retrieve top 20 chunks with the embedding model -> choose top 5 with the reranker model

def retrieve(query, k_initial=20, k_final=5):
    # Pasul A: Retrieval inițial (extrage mai multe fragmente)
    query_prefixed = "query: " + query
    q_emb = embedding_model.encode([query_prefixed])
    distances, indices = index.search(np.array(q_emb), k_initial)
    
    initial_chunks = [chunks[i] for i in indices[0]]

    if k_final == 0:  # 0 means skip reranking
        return initial_chunks


    # Pasul B: Reranking (Cross-Encoding)

    # Modelul compară perechi (întrebare, fragment) și dă un scor de relevanță
    sentence_combinations = [[query, chunk] for chunk in initial_chunks]
    rerank_scores = reranker_model.predict(sentence_combinations)
    
    # Pasul C: Sortăm fragmentele în funcție de scorul Reranker-ului
    reranked_results = sorted(
        list(zip(initial_chunks, rerank_scores)), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Returnăm doar primele k_final cele mai bune
    return [res[0] for res in reranked_results[:k_final]]



# QUESTION ANSWERING LLM

def ask_llm(question, context_chunks):
    context = "\n\n--- FRAGMENT NOU ---\n\n".join(context_chunks)

    prompt = f"""
Ești un asistent virtual oficial care ajută candidații cu informații despre admiterea la FMI UBB Cluj.
Trebuie să fii clar, concis și să răspunzi DOAR pe baza contextului oferit.

REGULI STRICTE:
1. Răspunde EXCLUSIV în limba română.
2. Folosește DOAR informațiile din secțiunea "Context". Nu folosi cunoștințele tale generale.
3. Dacă răspunsul nu se află în "Context", nu încerca să ghicești. Scrie EXACT următoarea propoziție: "Nu dețin această informație conform datelor extrase de pe site-ul oficial."
4. Nu inventa formule de calcul, date sau taxe.

Context:
{context}

Întrebare:
{question}

Răspuns:
"""

    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": "google/gemma-4-e4b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    )


    data = response.json()
    
    if "choices" not in data:
        print("Eroare primită de la LM Studio:", data)
        return "Eroare la generarea răspunsului."

    return data["choices"][0]["message"]["content"]



# SCORING SYSTEM LLM
def score_llm(question, context_chunks, answer, gold_answer):
    context = "\n\n--- FRAGMENT NOU ---\n\n".join(context_chunks)

    prompt = f"""
You are a ruthless and objective evaluator of a RAG (Retrieval-Augmented Generation) chatbot for university admissions.
Your task is to evaluate the chatbot's answer. The chatbot answered on a context it has been given.
You will receive the question, the chatbot's answer for it, the context based on which the chatbot has answered, and a gold answer.
Use the reference context to determine whether the chatbot hallucinated or not.
Use the gold answer to determine the factual correctness and the completeness of the chatbot's answer.

When the Golden Answer contains facts absent from the Reference Context, this
indicates a RETRIEVAL failure, not a hallucination — score no_hallucination based
strictly on consistency with the context, and score factual/completeness based on
the Golden Answer.

Question:
{question}

Chatbot Answer:
{answer}

Reference Context:
{context}

Gold Answer:
{gold_answer}

---
SCORING RUBRIC (Be extremely strict!):
1. Factual correctness (0-10): 
   - 10 = Perfectly aligned with the golden answer.
   - 5 = Partially correct, but contains errors.
   - 0 = The information is completely wrong or invented.
2. Completeness (0-10): Did the answer cover all the key facts present in the Golden Answer?
   - 10 = All key facts from the gold answer are present.
   - 5 = Some key facts missing.
   - 0 = Most key facts missing.
3. Clarity (0-10): Is the answer easy to read and in Romanian? (Score 0 if the answer is in English!).
4. Hallucinations penalty (0-10): 
   - 10 = NO hallucinations at all. Only used the context.
   - 0 = The model invented rules, formulas, or deadlines NOT present in the Reference Context.

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



# QA Model function
def answer_question(question, relevant_chunks):
    answer = ask_llm(question, relevant_chunks)
    return answer

# LLM-as-a-Judge Model function
def score_answer(question, relevant_chunks, answer, gold_answer):
    answer = score_llm(question, relevant_chunks, answer, gold_answer)
    return answer






# Compute the total score of the judge given scores.
# Returns the weighted sum of factual, complete, clarity and no_hallucination scores.
def compute_score(raw: str) -> dict:
    WEIGHTS = {
        "factual": 0.40,
        "complete": 0.25,
        "clarity": 0.15,
        "no_hallucination": 0.20,
    }

    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        data = json.loads(raw[start:end + 1])
    else:
        raise ValueError(f"Nu am putut extrage JSON din răspunsul judge-ului:\n{raw}")

    components = {}
    for key in WEIGHTS:
        val = float(data.get(key, 0))          
        components[key] = max(0.0, min(10.0, val))  

    final = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    data["final_score"] = round(final, 2)

    return str(data)



def answerAndScore(question, initialK=20, finalK=8):
    relevant_chunks = retrieve(question, initialK, finalK)
    answer = answer_question(question, relevant_chunks)
    score = compute_score(score_answer(question, relevant_chunks, answer, "This question does not have a gold answer. Use the reference context instead to validate the chatbot answer."))
    
    return answer, score