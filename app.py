import streamlit as st
from process import (
    extract_text_from_file,
    split_text,
    vectorize_chunks,
    split_json_if_large,
    send_to_webhook
)

st.title("🔍 Векторизация текста и файлов")

st.write("Загрузите файлы или введите текст вручную.")

# Загрузка файлов
uploaded_files = st.file_uploader("Загрузите файлы (TXT, PDF, DOCX, CSV)", accept_multiple_files=True)

# Ручной ввод текста
manual_text = st.text_area("Или введите текст вручную", "")

# Выбор модели векторизации
model_choice = st.selectbox("Выберите модель векторизации", ["openai", "yandex", "sentence_transformer"])

# Ввод Webhook URL
webhook_url = st.text_input("Webhook URL", "https://example.com/webhook")

# Слайдер для выбора максимального размера файла для отправки (от 1 до 10 МБ)
size_threshold_mb = st.slider("Выберите максимальный размер файла для отправки (МБ)", 1, 10, 1)
max_size_bytes = size_threshold_mb * 1024 * 1024

if st.button("🧠 Векторизировать"):
    full_text = ""
    # Если загружены файлы – обрабатываем каждый из них
    if uploaded_files:
        for file in uploaded_files:
            text = extract_text_from_file(file)
            if text:
                ext = file.name.lower().split('.')[-1]
                file_type = ext if ext in ["pdf", "docx"] else None
                chunks = split_text(text, file_type=file_type)
                full_text += "\n".join(chunks) + "\n"
    # Добавляем введённый вручную текст
    if manual_text:
        full_text += manual_text

    if not full_text.strip():
        st.error("Не найден текст для векторизации!")
    else:
        # Разбиваем текст на чанки
        chunks = split_text(full_text)
        st.write(f"Текст разбит на {len(chunks)} чанков")
        # Получаем эмбеддинги для каждого чанка
        vectors = vectorize_chunks(chunks, model_choice)
        # Формируем итоговый JSON
        json_data = {
            "status": "success",
            "model": model_choice,
            "original_text": full_text,
            "chunks": [
                {"chunk_id": i + 1, "text": chunk, "vector": vector}
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
        }
        # Если JSON больше заданного размера – разбиваем на части
        files_data = split_json_if_large(json_data, max_size_bytes)
        # Отправляем каждую часть на указанный Webhook
        for idx, data_part in enumerate(files_data):
            status_code, response_text = send_to_webhook(webhook_url, data_part)
            st.write(f"Часть {idx + 1} отправлена, статус: {status_code}")
        st.success("Данные успешно отправлены на обработку!")
