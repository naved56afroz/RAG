from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import oracledb
import time
import requests

app = FastAPI()

# ✅ Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://localhost:5500"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Oracle connection
connection = oracledb.connect(user="vector_user", password="Oracle_4U", dsn="localhost:1521/FREEPDB1")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    start = time.time()

    # Step 1: get relevant context from Oracle 23ai
    cur = connection.cursor()
    cur.execute("""
        SELECT MY_DATA
        FROM MY_TABLE
        ORDER BY vector_distance(v, (vector_embedding(ALL_MINILM_L12_V2 using :1 as data)))
        FETCH FIRST 1 ROWS ONLY
    """, [query.question])
    row = cur.fetchone()
    context = row[0] if row else "No relevant data found."

    # Step 2: Call Ollama running locally (Granite)
    system_prompt = (
        "You are Nava, an AI assistant trained to help users using context from an Oracle database. "
        "Be precise, helpful, and don’t invent facts. Only answer based on the context provided."
    )

    payload = {
        "model": "autopilot",  # or "granite-code:8b"
        "prompt": f"Context: {context}\n\nQuestion: {query.question}",
        "system": system_prompt,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        output = response.json().get("response", "Model did not return a response.")
    except Exception as e:
        output = f"Error calling Ollama: {str(e)}"

    end = time.time()
    return {
        "response": output,
        "time_taken": round(end - start, 2)
    }

