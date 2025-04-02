from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import psycopg2
from minio import Minio
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

# Initialize models
text_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
vision_model = SentenceTransformer('clip-ViT-B-32')
llm = pipeline("text-generation", model="elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")

# MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# PostgreSQL connection
conn = psycopg2.connect("postgres://postgres:postgres@postgres:5432/postgres")
cur = conn.cursor()

class Query(BaseModel):
    text: str

@app.post("/query")
async def handle_query(query: Query):
    try:
        # Get recent 3 minutes of data
        three_minutes_ago = datetime.now() - timedelta(minutes=3)
        
        # Search text embeddings
        cur.execute("""
            SELECT text_content 
            FROM text_embeddings 
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
            LIMIT 10
        """, (three_minutes_ago,))
        recent_texts = [row[0] for row in cur.fetchall()]
        
        # Search image embeddings
        cur.execute("""
            SELECT image_path 
            FROM image_embeddings 
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
            LIMIT 10
        """, (three_minutes_ago,))
        recent_images = [row[0] for row in cur.fetchall()]
        
        # Generate context
        context = f"直近のテキスト入力: {' '.join(recent_texts)}\n\n"
        context += f"直近の画像パス: {', '.join(recent_images)}\n\n"
        context += f"ユーザークエリ: {query.text}"
        
        # Generate response
        response = llm(context, max_length=200, do_sample=True)
        
        return {"response": response[0]['generated_text']}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
