import streamlit as st
import requests
import pdfplumber
import docx2txt
import pandas as pd
import json

# Импорт функции обработки данных из processing.py
from processing import process_text_and_files

st.title("🔍 Векторизация текста и файлов")
st.write("Загрузите файлы (TXT, PDF, DOCX, CSV) или введите текст вручную для векторизации.")

uploaded_files = st.file_uploader(
    "Выберите файлы для загрузки", 
    type=["txt", "pdf", "docx", "csv"], 
    accept_multiple_files=True
)
manual_text = st.text_area("Введите текст вручную", placeholder="Вставьте или введите текст здесь...")

selected_model = st.selectbox(
    "Выберите модель векторизации", 
    ["openai", "yandex", "sentence_transformer"]
)
webhook_url = st.text_input("Webhook URL", "https://example.com/webhook")

# Тумблер для выбора размера чанков
chunk_size_option = st.radio(
    "Выберите размер чанка:",
    ("500 КБ", "1 МБ", "2 МБ")
)

if chunk_size_option == "500 КБ":
    chunk_size = 500 * 1024
elif chunk_size_option == "1 МБ":
    chunk_size = 1 * 1024 * 1024
elif chunk_size_option == "2 МБ":
    chunk_size = 2 * 1024 * 1024

st.write(f"Выбранный размер чанка: {chunk_size_option} ({chunk_size} байт)")

if st.button("🧠 Векторизировать"):
    try:
        # Передаём новый параметр chunk_size в функцию обработки
        result = process_text_and_files(
            uploaded_files, manual_text, selected_model, webhook_url, chunk_size=chunk_size
        )
        if result:
            st.success("Данные успешно обработаны и отправлены на Webhook!")
        else:
            st.error("Произошла ошибка при обработке данных.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
