from Functionality.chatbotFunction import ask_chatbot
from Functionality.judgeFunction import judge_scores
import json
import pandas as pd



# QA Model function
def answer_question(question, relevant_chunks):
    answer = ask_chatbot(question, relevant_chunks)
    return answer

# LLM-as-a-Judge Model function
def score_answer(question, relevant_chunks, answer):
    answer = judge_scores(question, relevant_chunks, answer)
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
    return data


# retrieve: the retrieval function
# initialK & finalK: For the retrieval function
# question: ask one specific question
def answerAndScore(retrieve, initialK=20, finalK=8, question=None):
        relevant_chunks = retrieve(question, initialK, finalK)
        answer = answer_question(question, relevant_chunks)
        print(answer) 
        score = compute_score(score_answer(question, relevant_chunks, answer))
        print("\n\n",score)