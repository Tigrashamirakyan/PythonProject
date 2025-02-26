import streamlit as st
import requests
import pdfplumber
import docx2txt
import pandas as pd
import json

# Импорт функции обработки данных из process.py
from process import process_text_and_files

# Заголовок приложения
st.title("🔍 Векторизация текста и файлов")

# Краткое описание
st.write("Загрузите файлы (TXT, PDF, DOCX, CSV) или введите текст вручную для векторизации.")

# Загрузка файлов (поддерживаются файлы с расширениями: txt, pdf, docx, csv)
uploaded_files = st.file_uploader(
    "Выберите файлы для загрузки", 
    type=["txt", "pdf", "docx", "csv"], 
    accept_multiple_files=True
)

# Текст, вводимый вручную
manual_text = st.text_area("Введите текст вручную", placeholder="Вставьте или введите текст здесь...")

# Выбор модели векторизации
selected_model = st.selectbox(
    "Выберите модель векторизации", 
    ["openai", "yandex", "sentence_transformer"]
)

# Ввод URL для Webhook, куда будут отправлены результаты
webhook_url = st.text_input("Webhook URL", "https://example.com/webhook")

# Кнопка запуска процесса векторизации
if st.button("🧠 Векторизировать"):
    try:
        # Вызов функции, которая обрабатывает файлы и текст,
        # выполняет векторизацию, разбивку на чанки и отправку результата на указанный Webhook.
        result = process_text_and_files(uploaded_files, manual_text, selected_model, webhook_url)
        
        if result:
            st.success("Данные успешно обработаны и отправлены на Webhook!")
        else:
            st.error("Произошла ошибка при обработке данных.")
    except Exception as e:
        st.error(f"Ошибка: {e}")
