from Functionality.chatbotFunction import ask_chatbot
from Functionality.judgeFunction import judge_scores
import json
import pandas as pd
import time
import os



# QA Model function
def answer_question(question, relevant_chunks):
    answer = ask_chatbot(question, relevant_chunks)
    return answer

# LLM-as-a-Judge Model function
def score_answer(question, relevant_chunks, answer):
    answer = judge_scores(question, relevant_chunks, answer)
    return answer


def load_questions(path):
    # 1. Get the directory of THIS file (Functionality/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Go up one level to the project root directory
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    # 3. Build the absolute path straight to the target folder
    target_path = os.path.join(project_root, "Experimentation", "Experiments", path, "questions.json")
    
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)

        
def save_results(path, scores):
    df = pd.DataFrame(scores)
    
    # 1. Get the directory of THIS file (Functionality/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. Go up one level to the project root directory
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    # 3. Build the absolute path straight to the target folder
    target_path = os.path.join(project_root, "Experimentation", "Experiments", path, "answersAndScores.csv")

    df.to_csv(target_path, index=False, encoding="utf-8")



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
# path: the path to the experiment folder
def answerAndScore(retrieve, path=None, initialK=20, finalK=8):
    questions = load_questions(path)
    scores = []
    totalTime = 0
    
    ct = 1
    for q in questions:
            start = time.time()
            relevant_chunks = retrieve(q["question"], initialK, finalK)

            answer = answer_question(q["question"], relevant_chunks)
            print("Answered question ", ct) 

            score = compute_score(score_answer(q, relevant_chunks, answer))
            print("Scored answer ", ct) ; ct+=1
            scores.append({
                "question": q["question"],
                "answer": answer,
                "score": score
            })

            elapsed = int(time.time() - start) ; minutes = elapsed // 60 ; seconds = elapsed % 60
            print(f"Time: {minutes}m{seconds}s")
            totalTime += elapsed

            save_results(path, scores)
            
    print(f"Total time: {totalTime//60}m{totalTime%60}s")