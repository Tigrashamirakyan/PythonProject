import requests
import pdfplumber
import docx2txt
import openai
from sentence_transformers import SentenceTransformer
import pandas as pd

def split_text_into_chunks(text, chunk_size, overlap=50):
    """
    Разбивает текст на чанки заданного размера с перекрытием.
    Если текст меньше 500 символов, возвращается список из одного элемента.
    """
    if len(text) < 500:
        return [text]
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # смещение с перекрытием
    return chunks

def get_openai_embedding(text):
    """
    Получает векторное представление текста через API OpenAI.
    Требует предварительной установки openai.api_key.
    """
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

def get_yandex_embedding(text):
    """
    Заглушка для получения эмбеддинга через Yandex GPT.
    Здесь необходимо реализовать реальный вызов API, если он доступен.
    Пока возвращает список нулей длины 768.
    """
    return [0.0] * 768

def get_sentence_transformer_embedding(text):
    """
    Получает векторное представление текста с помощью SentenceTransformer.
    При каждом вызове модель инициализируется заново – для оптимизации можно создать модель один раз.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding.tolist()

def process_text_and_files(uploaded_files, manual_text, selected_model, webhook_url, chunk_size):
    """
    Обрабатывает загруженные файлы и текст:
      - Извлекает текст из файлов (TXT, PDF, DOCX, CSV)
      - Объединяет с текстом, введённым вручную
      - Разбивает полученный текст на чанки по указанному размеру
      - Выполняет векторизацию каждого чанка выбранной моделью
      - Формирует JSON-объект и отправляет его на указанный Webhook URL
    """
    all_text = ""

    # Обработка загруженных файлов
    if uploaded_files:
        for file in uploaded_files:
            filename = file.name.lower()
            if filename.endswith(".txt"):
                try:
                    text = file.read().decode("utf-8")
                    all_text += text + "\n"
                except Exception as e:
                    print(f"Ошибка чтения {file.name}: {e}")
            elif filename.endswith(".pdf"):
                try:
                    with pdfplumber.open(file) as pdf:
                        pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                        text = "\n".join(pages_text)
                        all_text += text + "\n"
                except Exception as e:
                    print(f"Ошибка обработки PDF {file.name}: {e}")
            elif filename.endswith(".docx"):
                try:
                    text = docx2txt.process(file)
                    all_text += text + "\n"
                except Exception as e:
                    print(f"Ошибка обработки DOCX {file.name}: {e}")
            elif filename.endswith(".csv"):
                try:
                    # Предположим, что CSV содержит текстовые данные
                    df = pd.read_csv(file)
                    # Соединяем все значения в строку
                    text = df.apply(lambda row: " ".join(row.values.astype(str)), axis=1).str.cat(sep="\n")
                    all_text += text + "\n"
                except Exception as e:
                    print(f"Ошибка обработки CSV {file.name}: {e}")

    # Добавляем введённый вручную текст
    if manual_text:
        all_text += manual_text + "\n"

    if not all_text.strip():
        raise ValueError("Нет текста для обработки.")

    # Разбиваем текст на чанки с использованием выбранного размера
    chunks = split_text_into_chunks(all_text, chunk_size)

    # Векторизация каждого чанка с выбранной моделью
    vectors = []
    for chunk in chunks:
        if selected_model == "openai":
            vector = get_openai_embedding(chunk)
        elif selected_model == "yandex":
            vector = get_yandex_embedding(chunk)
        elif selected_model == "sentence_transformer":
            vector = get_sentence_transformer_embedding(chunk)
        else:
            vector = None
        vectors.append(vector)

    # Формируем JSON-объект для отправки на Webhook
    data = {
        "status": "success",
        "model": selected_model,
        "original_text": all_text,
        "chunks": [
            {"chunk_id": i+1, "text": chunk, "vector": vec}
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
    }

    # Если размер данных превышает 1 МБ, можно реализовать логику разбиения на несколько запросов.
    # Здесь предполагается, что итоговый JSON не превышает ограничение.
    response = requests.post(webhook_url, json=data)
    return response.ok
