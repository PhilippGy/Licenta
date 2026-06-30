# LOAD
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import numpy as np
from pathlib import Path
from functools import partial

from .retrieval import retrieve
from .answerOneQuestion import answerAndScore
from .removeLogging import *




SCRIPT_DIR = Path(__file__).parent
CHUNKS_PATH = SCRIPT_DIR.parent / "Data" / "chunks.npy"
INDEX_PATH = SCRIPT_DIR.parent / "Data" / "index.faiss"

chunks = np.load(CHUNKS_PATH, allow_pickle=True)
index = faiss.read_index(str(INDEX_PATH))


embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
reranker_model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


# Retrieval & Reranking
# "partial()" makes "retrieval" be a local lambda function replacing the actual retrieve()
retrieval = partial(retrieve, embedding_model, index, chunks, reranker_model)


# Both the Chatbot and the Judge answer
def ask_question(question):
    answerAndScore(question=question, retrieve=retrieval)

