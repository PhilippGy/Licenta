from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # baza de date specializata in viteza. Daca ai 1 milion de fragmente de text, FAISS (Facebook AI Similarity Search) gaseste instant care dintre ele seamana cu intrebarea ta
import numpy as np  # FAISS lucreaza doar cu structuri de date de tip numpy



DATA_DIR = Path("Data/MD Files")
candidates = sorted(DATA_DIR.glob("admission_kb_*.md"))
if not candidates:
    raise FileNotFoundError("No admission_kb_*.md files found in Data/")

KNOWLEDGE_BASE = candidates[-1]   # the most recent admission_kb.md file

# USE AN OLDER KNOWLEDGE BASE INSTEAD:
# KNOWLEDGE_BASE = DATA_DIR / "admission_kb_YYYY_MM_DD.md"    



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

    # Adaugă prefixul obligatoriu pentru modelul E5 la indexare
    chunks_with_prefix = [f"passage: {c}" for c in chunks]
    embeddings = model.encode(chunks)

    # Creeaza un index (memorie rapida) care masoara distanta "L2" (distanta euclidiana, ca o linie dreapta pe harta) intre vectori
    index = faiss.IndexFlatL2(len(embeddings[0]))  # len(embeddings[0]) -> ii spune indexului cat de lung este un singur vector (pentru acest model in specific lungimea fiind de 384 de numere)
    index.add(np.array(embeddings))

    return index



def generate_chunks_and_embeddings():
    # Citire și aplicare
    with open(KNOWLEDGE_BASE, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean text
    text = re.sub(r' +', ' ', text) 

    chunks = refined_chunking(text)
    print(f"Generat {len(chunks)} chunks optimizați.")

    index = compute_embeddings(chunks)
    print("Embeddings calculate și index FAISS creat.")

    # SAVE
    SCRIPT_DIR = Path(__file__).parent
    np.save(f"{SCRIPT_DIR}/../chunks.npy", chunks)
    faiss.write_index(index, f"{SCRIPT_DIR}/../index.faiss")