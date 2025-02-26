import streamlit as st
from processing import process_text_files  # Импортируем функцию из processing.py

# Интерфейс Streamlit
st.title("Text Processing App")

# Загружаем файлы
uploaded_files = st.file_uploader("Upload text files", accept_multiple_files=True)
manual_text = st.text_area("Or enter text manually")
webhook_url = st.text_input("Enter webhook URL")
chunk_size = st.number_input("Enter chunk size", min_value=1, value=100)

# Модель (инициализируем SentenceTransformer как пример, если нужно)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Проверяем типы аргументов
st.write(f"uploaded_files: {type(uploaded_files)}")
st.write(f"manual_text: {type(manual_text)}")
st.write(f"model: {type(model)}")
st.write(f"webhook_url: {type(webhook_url)}")
st.write(f"chunk_size: {type(chunk_size)}")

# Обработка ошибок и кнопка для начала процесса
if st.button("Process"):
    try:
        success = process_text_files(uploaded_files, manual_text, model, webhook_url)
        st.success("Processing completed successfully!" if success else "Processing failed.")
    except Exception as e:
        st.error(f"Error occurred: {e}")
