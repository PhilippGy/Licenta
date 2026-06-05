# LOAD
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import numpy as np


chunks = np.load("Data/chunks.npy", allow_pickle=True)
index = faiss.read_index("Data/index.faiss")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')




# RETRIEVAL 
# retrieve top 20 chunks with the embedding model -> choose top 5 with the reranker model

def retrieve(query, k_initial=20, k_final=8):
    # Pasul A: Retrieval inițial (extrage mai multe fragmente)
    query_prefixed = "query: " + query
    q_emb = embedding_model.encode([query_prefixed])
    distances, indices = index.search(np.array(q_emb), k_initial)
    
    initial_chunks = [chunks[i] for i in indices[0]]
    
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
import requests

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
def score_llm(question, context_chunks, answer):
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

Return ONLY valid JSON format:
{{
  "factual": <int>,
  "complete": <int>,
  "clarity": <int>,
  "no_hallucination": <int>,
  "final_score": <float between 0 and 10>,
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
def score_answer(question, relevant_chunks, answer):
    answer = score_llm(question, relevant_chunks, answer)
    return answer





import tkinter as tk
from tkinter import font as tkfont
import threading
import time


def answerAndScore(
    initialK=20,
    finalK=5,
    question=None,
    ui_callbacks=None,          
):
    """
    Batch mode  (path != None, question == None) : unchanged — iterates over
                                                    questions, saves results.
    Single mode (path == None, question != None) : results are streamed to the
                                                    UI via ui_callbacks instead
                                                    of being printed.

    ui_callbacks keys (all callables):
        on_answer(text)   – called once with the full QA answer
        on_score(text)    – called once with the full judge verdict
        on_done()         – called when everything is finished
        on_error(msg)     – called on unexpected exceptions
    """

    
    cb = ui_callbacks or {}
    try:
        relevant_chunks = retrieve(question, initialK, finalK)

        answer = answer_question(question, relevant_chunks)
        if "on_answer" in cb:
                cb["on_answer"](answer)

        score = score_answer(question, relevant_chunks, answer)
        if "on_score" in cb:
            cb["on_score"](score)

        if "on_done" in cb:
            cb["on_done"]()

    except Exception as exc:
        if "on_error" in cb:
            cb["on_error"](str(exc))


# UI

BG          = "#0f1117"
PANEL       = "#1a1d27"
BORDER      = "#2a2d3d"
ACCENT      = "#7c6af7"          
ACCENT2     = "#4fd1c5"          
TEXT        = "#e2e8f0"
TEXT_DIM    = "#64748b"
ENTRY_BG    = "#151821"
BTN_ACTIVE  = "#6254d4"


TOKEN_DELAY = 28                 


class ChatbotUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("QA Chatbot  ·  LLM-as-a-Judge")
        self.root.configure(bg=BG)
        self.root.geometry("860x720")
        self.root.minsize(640, 500)

        self._setup_fonts()
        self._build_layout()

    def _setup_fonts(self):
        self.font_label  = tkfont.Font(family="Courier New", size=9,  weight="bold")
        self.font_input  = tkfont.Font(family="Courier New", size=11)
        self.font_answer = tkfont.Font(family="Courier New", size=10)
        self.font_title  = tkfont.Font(family="Courier New", size=13, weight="bold")
        self.font_btn    = tkfont.Font(family="Courier New", size=10, weight="bold")

    def _build_layout(self):
        root = self.root

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(
            header, text="◈  QA CHATBOT  +  JUDGE",
            font=self.font_title, fg=ACCENT, bg=BG,
        ).pack(side="left")

        q_outer, q_frame = self._panel(root, label="QUESTION")
        q_outer.pack(fill="x", padx=24, pady=(10, 0))

        self.question_text = tk.Text(
            q_frame, height=3, font=self.font_input,
            bg=ENTRY_BG, fg=TEXT, insertbackground=ACCENT,
            relief="flat", bd=0, padx=10, pady=8,
            wrap="word", highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self.question_text.pack(fill="x", padx=12, pady=(4, 8))
        self.question_text.bind("<Return>",    self._on_enter)
        self.question_text.bind("<Shift-Return>", lambda e: None)   

        btn_row = tk.Frame(q_frame, bg=PANEL)
        btn_row.pack(fill="x", padx=12, pady=(0, 10))

        self.submit_btn = tk.Button(
            btn_row, text="  ▶  ASK  ",
            font=self.font_btn, bg=ACCENT, fg="#ffffff",
            activebackground=BTN_ACTIVE, activeforeground="#ffffff",
            relief="flat", bd=0, cursor="hand2", padx=6, pady=4,
            command=self._submit,
        )
        self.submit_btn.pack(side="left")

        self.status_label = tk.Label(
            btn_row, text="", font=self.font_label,
            fg=TEXT_DIM, bg=PANEL,
        )
        self.status_label.pack(side="left", padx=14)

        self.clear_btn = tk.Button(
            btn_row, text="clear",
            font=self.font_label, bg=PANEL, fg=TEXT_DIM,
            activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, cursor="hand2",
            command=self._clear,
        )
        self.clear_btn.pack(side="right")

        ans_outer, ans_inner = self._panel(root, label="ANSWER")
        ans_outer.pack(fill="both", expand=True, padx=24, pady=(12, 0))

        self.answer_text = self._output_box(ans_inner, ACCENT)

        judge_outer, judge_inner = self._panel(root, label="JUDGE VERDICT")
        judge_outer.pack(fill="both", expand=True, padx=24, pady=(10, 20))

        self.judge_text = self._output_box(judge_inner, ACCENT2)

    def _panel(self, parent, label: str):
        """Returns (outer, inner). Caller packs outer; widgets go in inner."""
        outer = tk.Frame(parent, bg=BORDER, bd=0)
        inner = tk.Frame(outer, bg=PANEL, bd=0)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(
            inner, text=f"  {label}  ",
            font=self.font_label, fg=TEXT_DIM, bg=PANEL,
        ).pack(anchor="nw", padx=6, pady=(6, 0))
        return outer, inner

    def _output_box(self, parent: tk.Frame, cursor_color: str) -> tk.Text:
        box = tk.Text(
            parent, font=self.font_answer,
            bg=PANEL, fg=TEXT, insertbackground=cursor_color,
            relief="flat", bd=0, padx=12, pady=8,
            wrap="word", state="disabled",
            highlightthickness=0,
            spacing1=2, spacing3=2,
        )
        box.pack(fill="both", expand=True, padx=0, pady=(2, 8))
     
        box.tag_config("dim",   foreground=TEXT_DIM)
        box.tag_config("token", foreground=TEXT)
        return box

    def _on_enter(self, event):
        if event.state & 0x1:         
            return
        self._submit()
        return "break"                 

    def _submit(self):
        question = self.question_text.get("1.0", "end-1c").strip()
        if not question:
            return

        self._clear_outputs()
        self._set_ready(False)
        self._status("retrieving…")

        threading.Thread(
            target=self._run_pipeline,
            args=(question,),
            daemon=True,
        ).start()

    def _clear(self):
        self.question_text.delete("1.0", tk.END)
        self._clear_outputs()
        self._status("")

    def _clear_outputs(self):
        for box in (self.answer_text, self.judge_text):
            box.config(state="normal")
            box.delete("1.0", tk.END)
            box.config(state="disabled")

    def _set_ready(self, ready: bool):
        state = "normal" if ready else "disabled"
        self.submit_btn.config(state=state)

    def _status(self, msg: str):
        self.root.after(0, self.status_label.config, {"text": msg})

    def _run_pipeline(self, question: str):
        self._status("generating answer…")

        answer_holder = [None]
        score_holder  = [None]
        error_holder  = [None]
        done_event    = threading.Event()
        score_event   = threading.Event()

        def on_answer(text):
            answer_holder[0] = text
            
            self.root.after(0, self._type_into, self.answer_text, text,
                            0, score_event.set)         

        def on_score(text):
            score_event.wait()                          
            score_holder[0] = text
            self.root.after(0, self._type_into, self.judge_text, text,
                            0, done_event.set)

        def on_done():
            done_event.wait()
            self.root.after(0, self._finish)

        def on_error(msg):
            error_holder[0] = msg
            self.root.after(0, self._show_error, msg)

        answerAndScore(
            question=question,
            ui_callbacks={
                "on_answer": on_answer,
                "on_score":  on_score,
                "on_done":   on_done,
                "on_error":  on_error,
            },
        )

    def _finish(self):
        self._status("done ✓")
        self._set_ready(True)

    def _show_error(self, msg: str):
        self._status(f"error: {msg}")
        self._set_ready(True)

    def _type_into(self, box: tk.Text, text: str, index: int, done_cb=None):
        """
        Recursively schedules itself via root.after() to append one token
        at a time into *box*.  Calls done_cb() when the full text is written.
        """
        tokens = text.split(" ")       

        def _step(i: int):
            if i >= len(tokens):
                if done_cb:
                    done_cb()
                return
            word = tokens[i]
            sep  = " " if i < len(tokens) - 1 else ""
            box.config(state="normal")
            box.insert(tk.END, word + sep, "token")
            box.see(tk.END)
            box.config(state="disabled")
            self.root.after(TOKEN_DELAY, _step, i + 1)

        if index == 0:
            box.config(state="normal")
            box.delete("1.0", tk.END)
            box.config(state="disabled")

        _step(index)


def launch_ui():
    root = tk.Tk()
    app  = ChatbotUI(root)
    root.mainloop()

