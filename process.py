import os
import json
import requests
import openai
import pdfplumber
import docx2txt
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
YANDEX_GPT_API_KEY = os.getenv("YANDEX_GPT_API_KEY")  # Используйте по необходимости

# Инициализация модели Sentence Transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_file(file):
    """Извлекает текст из файла в зависимости от его типа."""
    filename = file.name.lower()
    if filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif filename.endswith(".docx"):
        return docx2txt.process(file)
    elif filename.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            return df.to_csv(index=False)
        except Exception:
            return file.read().decode("utf-8")
    else:
        return None


def chunk_by_length(text, min_size=256, max_size=512, overlap=50):
    """
    Разбивает текст на чанки длиной от min_size до max_size с перекрытием.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + max_size
        if end >= text_length:
            chunks.append(text[start:text_length])
            break
        else:
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # Небольшое перекрытие
    return chunks


def split_text(text, file_type=None):
    """
    Разбивает текст на чанки.
    Если тип файла pdf или docx – используется разбиение по абзацам,
    иначе – если текст менее 500 символов, возвращает как есть, 
    а если больше – разбивает на чанки заданной длины.
    """
    if file_type in ["pdf", "docx"]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        for p in paragraphs:
            if len(p) > 500:
                chunks.extend(chunk_by_length(p, min_size=256, max_size=512))
            else:
                chunks.append(p)
        return chunks
    else:
        if len(text) < 500:
            return [text]
        else:
            return chunk_by_length(text, min_size=256, max_size=512)


def get_openai_embedding(text):
    """Получает эмбеддинг через OpenAI API (text-embedding-ada-002)."""
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Ошибка OpenAI: {e}")
        return []


def get_sentence_transformer_embedding(text):
    """Получает эмбеддинг через Sentence Transformers."""
    return sentence_model.encode(text).tolist()


def get_yandex_gpt_embedding(text):
    """
    Получает эмбеддинг через Yandex GPT API.
    Для демонстрации используется placeholder (вы можете заменить реализацию вызовом API).
    """
    # Здесь можно реализовать вызов реального API, используя YANDEX_GPT_API_KEY
    return sentence_model.encode(text).tolist()


def vectorize_chunks(chunks, model_choice):
    """Проходит по чанкам и получает эмбеддинги выбранной моделью."""
    vectors = []
    for chunk in chunks:
        if model_choice == "openai":
            vector = get_openai_embedding(chunk)
        elif model_choice == "yandex":
            vector = get_yandex_gpt_embedding(chunk)
        elif model_choice == "sentence_transformer":
            vector = get_sentence_transformer_embedding(chunk)
        else:
            vector = []
        vectors.append(vector)
    return vectors


def split_json_if_large(data, max_size_bytes):
    """
    Если JSON-данные (в виде строки) превышают max_size_bytes, 
    то разбивает список чанков на несколько частей.
    """
    json_str = json.dumps(data, ensure_ascii=False)
    size_bytes = len(json_str.encode('utf-8'))
    if size_bytes <= max_size_bytes:
        return [data]
    else:
        base_data = {
            "status": data.get("status"),
            "model": data.get("model"),
            "original_text": data.get("original_text")
        }
        chunks = data.get("chunks", [])
        files = []
        current_chunks = []
        for chunk in chunks:
            current_chunks.append(chunk)
            temp_data = base_data.copy()
            temp_data["chunks"] = current_chunks
            temp_json = json.dumps(temp_data, ensure_ascii=False)
            if len(temp_json.encode('utf-8')) > max_size_bytes:
                # Если превышает, удаляем последний добавленный элемент и создаём новый файл
                current_chunks.pop()
                file_data = base_data.copy()
                file_data["chunks"] = current_chunks
                files.append(file_data)
                current_chunks = [chunk]
        if current_chunks:
            file_data = base_data.copy()
            file_data["chunks"] = current_chunks
            files.append(file_data)
        return files


def send_to_webhook(webhook_url, data):
    """Отправляет данные в формате JSON на указанный Webhook."""
    try:
        response = requests.post(webhook_url, json=data)
        return response.status_code, response.text
    except Exception as e:
        return None, str(e)
