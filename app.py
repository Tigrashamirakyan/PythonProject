import streamlit as st
import json
import requests
from process import process_files

st.set_page_config(page_title="🔍 Векторизация текста и файлов", layout="wide")

st.title("🔍 Векторизация текста и файлов")

# Заголовок
st.markdown("### 🚀 Выберите файлы и модель векторизации")

# Загрузка файлов
uploaded_files = st.file_uploader("📂 Загрузите файлы (TXT, PDF, DOCX, CSV)", accept_multiple_files=True)

# Ввод дополнительного текста
manual_text = st.text_area("📝 Введите текст вручную (необязательно)")

# Выбор модели
model = st.selectbox("🧠 Выберите модель векторизации", ["openai", "yandex", "sentence_transformer"])

# Ввод Webhook URL
webhook_url = st.text_input("🌐 Введите Webhook URL", "https://example.com/webhook")

if st.button("🧠 Векторизировать"):
    if (uploaded_files or manual_text.strip()) and webhook_url:
        st.info("⏳ Обработка файлов и текста...")

        # Запускаем обработку файлов и текста
        success = process_files(uploaded_files, manual_text, model, webhook_url)

        if success:
            st.success("✅ Результаты успешно отправлены!")
        else:
            st.error("❌ Ошибка при обработке или отправке данных!")

    else:
        st.warning("⚠ Пожалуйста, загрузите файлы, введите текст или укажите Webhook URL!")