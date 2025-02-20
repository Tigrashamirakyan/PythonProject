import streamlit as st
import requests
from process import process_text_files

st.set_page_config(page_title="🔍 Векторизация текста и файлов", layout="wide")

st.title("🔍 Векторизация текста и файлов")

# Загрузка файлов
uploaded_files = st.file_uploader("📂 Загрузите файлы (TXT, PDF, DOCX)", accept_multiple_files=True)

# Ввод текста вручную
manual_text = st.text_area("✍ Введите текст вручную (если не загружаете файлы)")

# Выбор модели
model = st.selectbox("🧠 Выберите модель векторизации", ["openai", "yandex", "sentence_transformer"])

# Ввод Webhook URL
webhook_url = st.text_input("🌐 Введите Webhook URL", "https://example.com/webhook")

# Выбор размера чанков (по умолчанию 1 МБ)
chunk_size_mb = st.slider("Размер чанков (MB)", min_value=1, max_value=10, value=1)
chunk_size = chunk_size_mb * 1024 * 1024  # преобразуем в байты

if st.button("🧠 Векторизировать"):
    if (uploaded_files or manual_text) and webhook_url:
        st.info("⏳ Обработка данных...")

        # Обрабатываем файлы и текст
        success = process_text_files(uploaded_files, manual_text, model, webhook_url, chunk_size)

        if success:
            st.success("✅ Данные успешно обработаны и отправлены!")
        else:
            st.error("❌ Ошибка при обработке данных!")
    else:
        st.warning("⚠ Загрузите файлы или введите текст и укажите Webhook URL!")
