from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from programFunctionality import answerAndScore



app = FastAPI(title="Admitere FMI UBB RAG API")

# Configurare CORS (CRITIC pentru momentul în care vei folosi ngrok)
# Permitem orice origine "*" pentru a nu avea probleme când expui aplicația pe internet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structura datelor așteptate de la frontend
class QuestionRequest(BaseModel):
    question: str

# Endpoint pentru a servi pagina de frontend
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

# Endpoint-ul principal care face legătura cu logica ta
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        # Apelăm funcția ta RAG
        answer, score = answerAndScore(req.question)
        return {
            "answer": answer,
            "score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))