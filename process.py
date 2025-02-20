import os
import requests
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Загрузка переменных окружения
load_dotenv()

# API-ключи
openai.api_key = os.getenv("OPENAI_API_KEY")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

# URL Yandex GPT
YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# Модель Sentence Transformers
st_model = SentenceTransformer('all-MiniLM-L6-v2')


# 1. Извлечение текста из файлов
def extract_text_from_file(file):
    if file.name.endswith(".txt") or file.name.endswith(".csv"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif file.name.endswith(".docx"):
        from docx import Document
        document = Document(file)
        return "\n".join([para.text for para in document.paragraphs])
    return None


# 2. Разбиение текста на чанки
def split_text(text, max_chunk=512, overlap=50):
    text = text.strip()
    if len(text) <= 500:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk
        chunks.append(text[start:end])
        start = start + max_chunk - overlap
    return chunks


# 3. Векторизация текста
def get_openai_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        print("Ошибка OpenAI:", e)
        return None


def get_sentence_transformer_embedding(text):
    try:
        return st_model.encode(text).tolist()
    except Exception as e:
        print("Ошибка Sentence Transformers:", e)
        return None


def get_yandex_embedding(text):
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "yandexgpt",
        "input": text
    }

    try:
        response = requests.post(YANDEX_GPT_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("embedding")
        else:
            print("Ошибка Yandex GPT:", response.text)
            return None
    except Exception as e:
        print("Ошибка при вызове Yandex GPT:", e)
        return None


# 4. Отправка данных через Webhook
def send_to_webhook(webhook_url, model_name, chunks, vectors):
    data = {
        "status": "success",
        "model": model_name,
        "chunks": [{"chunk_id": i + 1, "text": chunk, "vector": vector}
                   for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
    }

    try:
        response = requests.post(webhook_url, json=data)
        return response.status_code == 200
    except Exception as e:
        print("Ошибка при отправке Webhook:", e)
        return False


# 5. Основная функция обработки
def process_text_files(uploaded_files, manual_text, model_name, webhook_url):
    all_text = ""

    # Извлекаем текст из файлов
    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_file(file)
            if text:
                all_text += text + "\n"

    # Добавляем текст, введённый вручную
    if manual_text:
        all_text += manual_text

    if not all_text.strip():
        print("Нет текста для обработки.")
        return False

    # Разбиваем текст на чанки
    chunks = split_text(all_text)

    # Векторизуем каждый чанк
    vectors = []
    for chunk in chunks:
        if model_name == "openai":
            vector = get_openai_embedding(chunk)
        elif model_name == "sentence_transformer":
            vector = get_sentence_transformer_embedding(chunk)
        elif model_name == "yandex":
            vector = get_yandex_embedding(chunk)
        else:
            print("Неизвестная модель:", model_name)
            vector = None

        if vector is None:
            print("Ошибка получения вектора для чанка:", chunk[:30])
            continue

        vectors.append(vector)

    if len(chunks) != len(vectors):
        print("Предупреждение: число чанков не совпадает с числом векторов.")

    return send_to_webhook(webhook_url, model_name, chunks, vectors)
