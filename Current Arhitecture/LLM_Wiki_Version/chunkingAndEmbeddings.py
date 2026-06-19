from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # baza de date specializata in viteza. Daca ai 1 milion de fragmente de text, FAISS (Facebook AI Similarity Search) gaseste instant care dintre ele seamana cu intrebarea ta
import numpy as np  # FAISS lucreaza doar cu structuri de date de tip numpy



def refined_chunking(text, chunk_size=1000, chunk_overlap=200):
    # Separatori ordonați după importanță (secțiune -> titlu -> paragraf -> linie -> frază)
    separators = ["\n---\n", "\n## ", "\n\n", "\n", ". ", " "]
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # 1. Stabilim limita teoretică
        end = min(start + chunk_size, text_len)
        
        # 2. Căutăm cel mai bun separator în fereastra curentă
        if end < text_len:
            chosen_end = end
            for sep in separators:
                # Căutăm separatorul de la dreapta la stânga pentru a maximiza dimensiunea chunk-ului
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    chosen_end = pos + len(sep)
                    break
            end = chosen_end

        # 3. Extragem chunk-ul
        chunk = text[start:end].strip()
        if len(chunk) > 20: 
            chunks.append(chunk)

        # Calculăm unde ar trebui să înceapă următorul chunk
        new_start = end - chunk_overlap
        
        # Dacă noul start nu avansează față de cel vechi, forțăm trecerea la 'end'
        # Asta se întâmplă când lungimea textului găsit e mai mică decât overlap-ul
        if new_start <= start:
            start = end
        else:
            start = new_start
            
    return chunks


# EMBEDDING

def compute_embeddings(chunks):
    model = SentenceTransformer("intfloat/multilingual-e5-small")

    chunks_with_prefix = [f"passage: {c}" for c in chunks]
    embeddings = np.asarray(model.encode(chunks_with_prefix, normalize_embeddings=True), dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index





# Wiki snapshots in dated subfolders, e.g. Data/Wiki/2026_06_15/Taxe.md
# (same dating idea as the admission_kb_YYYY_MM_DD files).
WIKI_ROOT = Path("Data/Wiki")     # <-- set to your wiki root
EXCLUDE = {"Index.md"}            # navigation only, no answerable content

def _resolve_wiki_dir(knowledgeBase):
    if knowledgeBase == "Most recent":    # find the most recent Wiki folder
        snapshots = sorted(p for p in WIKI_ROOT.iterdir() if p.is_dir())
        if not snapshots:
            raise FileNotFoundError(f"No wiki snapshots in '{WIKI_ROOT}'.")
        return snapshots[-1]
    return WIKI_ROOT / knowledgeBase

def clean_wiki(text):
    # drop trailing metadata sections (provenance/navigation, not content)
    text = re.split(r"\n## (?:Relationships|Source Notes)\b", text)[0]
    # [[Page|label]] -> label ; [[Page]] -> Page
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return re.sub(r" +", " ", text).strip()


def generate_chunks_and_embeddings(knowledgeBase="Most recent"):
    wiki_dir = _resolve_wiki_dir(knowledgeBase)
    files = sorted(p for p in wiki_dir.glob("*.md") if p.name not in EXCLUDE)
    if not files:
        raise FileNotFoundError(f"No .md files in '{wiki_dir}'.")
    print("Current wiki snapshot:", wiki_dir)

    all_chunks = []
    for path in files:
        source = path.stem.replace("-", " ")          # "Criterii-Admitere" -> "Criterii Admitere"
        text = clean_wiki(path.read_text(encoding="utf-8"))
        for c in refined_chunking(text):              # chunk each file independently
            all_chunks.append(f"[{source}]\n{c}")     # tag every chunk with its page
        print(f"  {path.name}")

    print(f"Generat {len(all_chunks)} chunks din {len(files)} fișiere.")

    index = compute_embeddings(all_chunks)
    print("Embeddings calculate și index FAISS creat.")

    SCRIPT_DIR = Path("Data") 
    np.save(SCRIPT_DIR / "chunks.npy", all_chunks)
    faiss.write_index(index, str(SCRIPT_DIR / "index.faiss"))