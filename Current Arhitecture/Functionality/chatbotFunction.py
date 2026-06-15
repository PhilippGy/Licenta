# QUESTION ANSWERING LLM
import requests

def ask_chatbot(question, context_chunks):
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