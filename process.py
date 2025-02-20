import os
import json
import requests
import pdfplumber
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# ------------------------
# 1. Извлечение текста из файлов
# ------------------------
def extract_text_from_file(file):
    """
    Извлекает текст из файла.
    Поддерживаются форматы: .txt, .csv, .pdf, .docx
    """
    if file.name.endswith(".txt") or file.name.endswith(".csv"):
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            print("Ошибка при чтении файла TXT/CSV:", e)
            return None
    elif file.name.endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print("Ошибка при обработке PDF:", e)
            return None
    elif file.name.endswith(".docx"):
        try:
            from docx import Document
            document = Document(file)
            return "\n".join([para.text for para in document.paragraphs])
        except Exception as e:
            print("Ошибка при обработке DOCX:", e)
            return None
    else:
        return None

# ------------------------
# 2. Разбиение текста на чанки
# ------------------------
def split_text(text, min_chunk=256, max_chunk=512, overlap=50):
    """
    Разбивает текст на чанки длиной до max_chunk символов.
    Если текст меньше 500 символов, возвращает его целиком.
    Перекрытие (overlap) помогает сохранить контекст между чанками.
    """
    text = text.strip()
    if len(text) <= 500:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + max_chunk - overlap
    return chunks

def split_text_by_paragraphs(text):
    """
    Разбивает текст на абзацы.
    Удаляет пустые строки.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs

# ------------------------
# 3. Векторизация текста
# ------------------------

# 3.1. Векторизация через OpenAI
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_embedding(text):
    """
    Получает эмбеддинг для текста через API OpenAI.
    Модель: text-embedding-ada-002.
    """
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response["data"][0]["embedding"]
    except Exception as e:
        print("Ошибка при вызове OpenAI:", e)
        return None

# 3.2. Векторизация через Sentence Transformers
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_transformer_embedding(text):
    """
    Получает эмбеддинг текста через Sentence Transformers.
    """
    try:
        vector = st_model.encode(text)
        return vector.tolist()
    except Exception as e:
        print("Ошибка при использовании Sentence Transformers:", e)
        return None

# 3.3. Векторизация через Yandex GPT (пример, требует адаптации под документацию)
def get_yandex_embedding(text):
    """
    Пример получения эмбеддинга через API Yandex GPT.
    """
    try:
        response = requests.post("https://api.yandex/gpt/embedding", json={"text": text})
        if response.status_code == 200:
            return response.json().get("embedding")
        else:
            print("Ошибка Yandex GPT:", response.text)
            return None
    except Exception as e:
        print("Ошибка при вызове Yandex GPT:", e)
        return None

# ------------------------
# 4. Отправка данных через Webhook
# ------------------------
def send_to_webhook(webhook_url, model_name, original_text, chunks, vectors):
    """
    Формирует JSON-объект и отправляет его на указанный Webhook.
    Структура JSON:
    {
      "status": "success",
      "model": <выбранная модель>,
      "original_text": <объединённый текст>,
      "chunks": [
          { "chunk_id": 1, "text": <текст чанка>, "vector": [числа...] },
          ...
      ]
    }
    """
    chunk_data = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_data.append({
            "chunk_id": i + 1,
            "text": chunk,
            "vector": vector
        })

    data = {
        "status": "success",
        "model": model_name,
        "original_text": original_text,
        "chunks": chunk_data
    }

    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 200:
            return True
        else:
            print("Ошибка при отправке Webhook:", response.text)
            return False
    except Exception as e:
        print("Ошибка при отправке данных на Webhook:", e)
        return False

# ------------------------
# 5. Основная функция обработки
# ------------------------
def process_files(uploaded_files, manual_text, model_name, webhook_url):
    """
    Обрабатывает файлы и/или текст:
      1. Извлекает текст из загруженных файлов.
      2. Добавляет текст, введённый вручную.
      3. Разбивает объединённый текст на чанки.
      4. Получает эмбеддинги для каждого чанка выбранной моделью.
      5. Отправляет итоговый JSON на Webhook.

    Возвращает True, если отправка прошла успешно, иначе False.
    """
    all_text = ""

    # Извлекаем текст из загруженных файлов
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

    # Разбиваем текст: если есть двойные переводы строк, используем абзацы, иначе по длине
    if "\n\n" in all_text:
        chunks = split_text_by_paragraphs(all_text)
    else:
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
            print("Не удалось получить вектор для чанка:", chunk[:30])
            continue  # Можно также вернуть False, если требуется строгая обработка
        vectors.append(vector)

    if len(chunks) != len(vectors):
        print("Предупреждение: число чанков не совпадает с числом векторов.")

    success = send_to_webhook(webhook_url, model_name, all_text, chunks, vectors)
    return success