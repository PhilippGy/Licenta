# Changes:
# - Previously, we were directly using the output of the crawler .
# - The output has been transformed into a markdown file which is much better structured.
# - The chunks will now consist of the md text.


import re
import numpy as np
import faiss  



def refined_chunking(text, chunk_size, chunk_overlap):
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

# Citire și aplicare
def apply_chunking(chunk_size, chunk_overlap):
    with open('Data/MD file/admission_kb.md', 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean text
    text = re.sub(r' +', ' ', text) 

    chunks = refined_chunking(text, chunk_size, chunk_overlap)
    print(f"Generat {len(chunks)} chunks optimizați.")
    return chunks



# EMBEDDING (neschimbat)
def embeddings(chunks, model):
    # Adaugă prefixul obligatoriu pentru modelul E5 la indexare
    chunks_with_prefix = [f"passage: {c}" for c in chunks]

    embs = model.encode(chunks_with_prefix, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])   # inner product pe vectori normalizați
    index.add(embs)

    return index


# SAVE
def save(index, chunks):
    np.save("Data/chunks.npy", chunks)
    faiss.write_index(index, "Data/index.faiss")



def computeChunksAndEmbeddings(chunk_size=1000, chunk_overlap=200, model=None):
    chunks = apply_chunking(chunk_size, chunk_overlap)
    index = embeddings(chunks, model)
    save(index, chunks)
