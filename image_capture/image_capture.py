import cv2
import time
from minio import Minio
from minio.error import S3Error
import psycopg2
import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP model for image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# MinIO client setup
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# PostgreSQL connection
conn = psycopg2.connect("postgres://postgres:postgres@postgres:5432/postgres")
cur = conn.cursor()

# Create bucket if not exists
try:
    if not minio_client.bucket_exists("images"):
        minio_client.make_bucket("images")
except S3Error as err:
    print(f"MinIO error: {err}")

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS image_embeddings (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def capture_and_process():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        timestamp = int(time.time())
        image_name = f"image_{timestamp}.jpg"
        
        # Save image to MinIO
        _, buffer = cv2.imencode('.jpg', frame)
        minio_client.put_object(
            "images",
            image_name,
            buffer.tobytes(),
            len(buffer.tobytes()),
            content_type="image/jpeg"
        )
        
        # Generate CLIP embedding
        image = transforms.ToPILImage()(frame)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs[0].cpu().numpy().tolist()
        
        # Save to PostgreSQL
        cur.execute(
            "INSERT INTO image_embeddings (image_path, embedding) VALUES (%s, %s)",
            (image_name, embedding)
        )
        conn.commit()
        
        time.sleep(1)  # Capture every 1 second

if __name__ == "__main__":
    capture_and_process()
