import streamlit as st

# Импортируй свою функцию
from your_module import process_text_files  # Убедись, что файл импортируется правильно

# Интерфейс Streamlit
st.title("Text Processing App")

uploaded_files = st.file_uploader("Upload text files", accept_multiple_files=True)
manual_text = st.text_area("Or enter text manually")
webhook_url = st.text_input("Enter webhook URL")
chunk_size = st.number_input("Enter chunk size", min_value=1, value=100)

# Загрузка модели (замени на свою)
model = None  # Здесь должен быть объект модели, если ты используешь нейросеть

# Проверяем типы аргументов
st.write(f"uploaded_files: {type(uploaded_files)}")
st.write(f"manual_text: {type(manual_text)}")
st.write(f"model: {type(model)}")
st.write(f"webhook_url: {type(webhook_url)}")
st.write(f"chunk_size: {type(chunk_size)}")

# Добавляем обработку ошибок
if st.button("Process"):
    try:
        success = process_text_files(uploaded_files, manual_text, model, webhook_url, chunk_size)
        st.success("Processing completed successfully!" if success else "Processing failed.")
    except Exception as e:
        st.error(f"Error occurred: {e}")
