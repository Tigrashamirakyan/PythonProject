import io
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer

def extract_text_from_file(file):
    """Извлекает текст из загруженного файла."""
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return None

def chunk_text(text, chunk_size=512, overlap=50):
    """Разбивает текст на чанки с перекрытием."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Перекрытие между чанками
    return chunks

def vectorize_text(chunks, model_name):
    """Загружает модель и векторизует чанки текста."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return [model.encode(chunk).tolist() for chunk in chunks]

def send_to_webhook(webhook_url, model_name, original_text, chunks, vectors):
    """Формирует JSON и отправляет на Webhook."""
    data = {
        "status": "success",
        "model": model_name,
        "original_text": original_text,
        "chunks": [{"chunk_id": i+1, "text": chunk, "vector": vector}
                   for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
    }
    
    response = requests.post(webhook_url, json=data)
    if response.status_code != 200:
        print(f"Ошибка Webhook: {response.status_code}, {response.text}")
    return response.status_code == 200

def process_text_and_files(uploaded_files, manual_text, model_name, webhook_url, chunk_size):
    """Обрабатывает загруженные файлы и введённый текст."""
    all_text = manual_text.strip()

    if uploaded_files:
        for file in uploaded_files:
            extracted_text = extract_text_from_file(file)
            if extracted_text:
                all_text += "\n" + extracted_text

    if not all_text:
        return False  # Нет текста для обработки

    chunks = chunk_text(all_text, chunk_size)
    vectors = vectorize_text(chunks, model_name)
    
    return send_to_webhook(webhook_url, model_name, all_text, chunks, vectors)
