import pyaudio
import wave
import time
import speech_recognition as sr
import psycopg2
from sentence_transformers import SentenceTransformer

# Initialize sentence transformer model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# PostgreSQL connection
conn = psycopg2.connect("postgres://postgres:postgres@postgres:5432/postgres")
cur = conn.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS text_embeddings (
    id SERIAL PRIMARY KEY,
    text_content TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

def record_and_process():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("音声入力を待機中...")
        
        while True:
            try:
                print("話してください...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                # Japanese speech recognition
                text = recognizer.recognize_google(audio, language='ja-JP')
                print(f"認識結果: {text}")
                
                # Generate text embedding
                embedding = model.encode(text).tolist()
                
                # Save to PostgreSQL
                cur.execute(
                    "INSERT INTO text_embeddings (text_content, embedding) VALUES (%s, %s)",
                    (text, embedding)
                )
                conn.commit()
                
            except sr.WaitTimeoutError:
                print("タイムアウト: 音声入力がありません")
                continue
            except sr.UnknownValueError:
                print("音声を認識できませんでした")
                continue
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                continue

if __name__ == "__main__":
    record_and_process()
