import streamlit as st
import requests
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Добавляем путь
from process import process_text_and_files

st.title("🔍 Векторизация текста и файлов")

uploaded_files = st.file_uploader("Загрузите файлы (TXT, PDF)", accept_multiple_files=True)
manual_text = st.text_area("Введите текст вручную")

model = st.selectbox("Выберите модель векторизации", ["openai", "yandex", "sentence_transformer"])
webhook_url = st.text_input("Webhook URL", "https://example.com/webhook")
chunk_size = st.slider("Размер чанков (в символах)", min_value=128, max_value=2048, step=128, value=512)

if st.button("🧠 Векторизировать"):
    if not webhook_url.startswith("http"):
        st.error("Введите корректный Webhook URL!")
    else:
        success = process_text_and_files(uploaded_files, manual_text, model, webhook_url, chunk_size)
        if success:
            st.success("Файлы успешно векторизованы и отправлены!")
        else:
            st.error("Ошибка при обработке данных!")
