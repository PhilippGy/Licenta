# Concept:
# A frontier model converts the raw data into one structured markdown file.

"""
Crawled university-admission .txt files -> one clean, structured Markdown knowledge base.

Strategy: a single API call. The full corpus (~30-50k tokens) fits in context with
room to spare, and the consolidated output is small (~3-4k tokens). 
One call lets the model see everything at once, which gives the best possible
deduplication. 
"""

import os
from pathlib import Path
from openai import OpenAI
from datetime import date


SCRIPT_DIR = Path(__file__).parent
DEFAULT_FOLDER_PATH = SCRIPT_DIR / "Crawled_documents"
OUTPUT_PATH = SCRIPT_DIR / f"../../MD Files/admission_kb_{date.today():%Y_%m_%d}.md"


# Soft guard; the check only fires if the input folder grows past a single model's context window.
TOKEN_WARN_THRESHOLD = 110_000


# --- API setup ---
api_key_str = os.environ.get("OPENAI_API_KEY")
if not api_key_str:
    with open(SCRIPT_DIR / "API_KEY.txt", "r", encoding="utf-8") as f:
        api_key_str = f.read().strip()
client = OpenAI(api_key=api_key_str)


# --- 1. Load the whole corpus (no batching) ---
def load_corpus(folder_name=DEFAULT_FOLDER_PATH):
    folder = Path(folder_name)
    if not folder.exists():
        raise FileNotFoundError(f"The directory '{folder}' was not found.")

    files = sorted(folder.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in '{folder}'.")

    wrapped = []
    for i, fp in enumerate(files, start=1):
        content = fp.read_text(encoding="utf-8").strip()
        wrapped.append(
            f"<DOCUMENT index='{i}' filename='{fp.name}'>\n{content}\n</DOCUMENT>"
        )
    corpus = "\n\n\n\n".join(wrapped)

    # Conservative token estimate (~3 chars/token for Romanian).
    est_tokens = len(corpus) // 3
    print(f"Loaded {len(files)} files, {len(corpus)} chars (~{est_tokens} tokens)")
    if est_tokens > TOKEN_WARN_THRESHOLD:
        print(f"  WARNING: ~{est_tokens} tokens may exceed the model context window. "
              f"If so, you'll need to chunk the input.")
    return corpus


# --- 2. Single call: corpus -> consolidated Markdown ---
SYSTEM_PROMPT = """You are an expert information extraction and document structuring system.

## Task
You will be given a batch of raw text documents (web pages, PDF extracts, or crawled content) related to university admission information for the Faculty of Mathematics and Computer Science (FMI), Babeș-Bolyai University (UBB Cluj). Your job is to transform this noisy, unstructured batch into ONE clean, structured, AI-ready Markdown knowledge base that will be used as the sole context for a question-answering chatbot.

## Important context
- The input documents may contain overlapping, duplicated, or partially related information.
- Each document may refer to different topics such as: admissions rules, fees, calendar, required documents, exam regulations, or contest rules.
- Organize information semantically, NOT by original document boundaries.
- Do NOT hallucinate or add external information.

## Preservation rules (priority: completeness over brevity)
- Preserve EVERY fact. A downstream chatbot can only answer what is present here; a missing fact becomes an unanswerable question or a hallucination.
- Preserve the CONDITIONS and SCOPE attached to every fact. If a fee, deadline, rule, or document applies only to a specific program, candidate category, year, or situation, state that qualifier alongside the fact. Never present a conditional fact as if it were universal.
- Keep exact values VERBATIM: amounts, dates, deadlines, percentages, regulation numbers, official document names, program names, and URLs. Do not paraphrase, round, or normalize these.
- Enumerate fully. Lists of required documents, steps, criteria, or exam components must be reproduced item by item, not summarized into a count or a sample.
- Make each fact self-contained: a reader (or retrieval system) seeing one section in isolation should understand it without needing another section.

## Deduplication rules
- Remove a duplicate ONLY when two statements are truly identical in meaning AND scope.
- If two statements look similar but differ in any condition (different program, category, year, amount, date), keep BOTH — they are different facts, not duplicates.
- If two sources conflict (e.g. different fees or dates for the same thing), keep both and flag the conflict explicitly; do not silently choose one.

## Output requirements
Produce a SINGLE Markdown document organized into coherent topic sections, using headings (#, ##, ###). Encouraged top-level topics (use those that apply; merge/rename as content requires):
- Prezentare generală / Admission rules
- Calendar
- Taxe (fees)
- Acte necesare (required documents)
- Regulament probe / examene (exam regulations)
- Reguli concurs (contest rules)

## Output rules
- Output ONLY the Markdown document. No explanations or preamble.
- Cut crawl boilerplate (menus, cookie banners, repeated footers, marketing filler) — this is noise, not detail.
- Favor explicit, factual phrasing over prose. Brevity in wording is fine; loss of facts or conditions is not.
"""


def build_knowledge_base(folder=DEFAULT_FOLDER_PATH, out_path=OUTPUT_PATH, max_rounds=4):
    corpus = load_corpus(folder)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": corpus},
    ]

    full = ""
    for r in range(max_rounds):
        response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages, temperature=0)
        choice = response.choices[0]
        full += choice.message.content
        if choice.finish_reason != "length":
            break
        messages.append({"role": "assistant", "content": choice.message.content})
        messages.append({"role": "user", "content": "Continue exactly where you stopped. Do not repeat anything."})
        print(f"  output hit length cap, continuing (round {r + 2})...")

    Path(out_path).write_text(full.strip(), encoding="utf-8")
    print("LLM CREATED THE MARKDOWN FILE")
    print(f"Wrote {out_path} ({len(full)} chars)")
    return out_path

