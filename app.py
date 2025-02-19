import streamlit as st
import json
import requests
from process import process_text_files

st.set_page_config(page_title="🔍 Векторизация текста и файлов", layout="wide")

st.title("🔍 Векторизация текста и файлов")

# Заголовок
st.markdown("### 🚀 Выберите файлы и модель векторизации")

# Загрузка файлов
uploaded_files = st.file_uploader("📂 Загрузите файлы (TXT, PDF)", accept_multiple_files=True)

# Выбор модели
model = st.selectbox("🧠 Выберите модель векторизации", ["openai", "yandex", "sentence_transformer"])

# Ввод Webhook URL
webhook_url = st.text_input("🌐 Введите Webhook URL", "https://example.com/webhook")

if st.button("🧠 Векторизировать"):
    if uploaded_files and webhook_url:
        st.info("⏳ Обработка файлов...")

        # Обрабатываем файлы
        chunks, vectors = process_text_files(uploaded_files, model)

        # Формируем JSON
        data = {
            "status": "success",
            "model": model,
            "chunks": [{"chunk_id": i+1, "text": chunk, "vector": vector}
                       for i, (chunk, vector) in enumerate(zip(chunks, vectors))]
        }

        # Отправляем Webhook
        response = requests.post(webhook_url, json=data)

        if response.status_code == 200:
            st.success("✅ Результаты успешно отправлены!")
        else:
            st.error("❌ Ошибка при отправке данных!")

    else:
        st.warning("⚠ Пожалуйста, загрузите файлы и укажите Webhook URL!")