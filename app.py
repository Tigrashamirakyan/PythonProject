# Импортируем библиотеку streamlit для создания веб-интерфейса
import streamlit as st
# Импортируем функцию обработки из файла process.py
from process import process_files

# Заголовок приложения
st.title("🔍 Векторизация текста и файлов")

# 1. Загрузка файлов
# Пользователь может загрузить один или несколько файлов (поддерживаемые типы: txt, pdf, docx, csv)
uploaded_files = st.file_uploader(
    "Загрузите файлы (TXT, PDF, DOCX, CSV)",
    type=['txt', 'pdf', 'docx', 'csv'],
    accept_multiple_files=True
)

# 2. Ввод текста вручную
manual_text = st.text_area("Или введите текст вручную")

# 3. Выбор модели векторизации
# Список моделей – OpenAI, Yandex и Sentence Transformer
model = st.selectbox("Выберите модель векторизации", ["openai", "yandex", "sentence_transformer"])

# 4. Ввод Webhook URL
# Этот URL будет использоваться для отправки результатов в виде JSON
webhook_url = st.text_input("Webhook URL", "https://example.com/webhook")

# 5. Кнопка для запуска процесса
if st.button("🧠 Векторизировать"):
    # Проверяем, что пользователь загрузил хотя бы один файл или ввёл текст
    if not uploaded_files and not manual_text:
        st.error("Пожалуйста, загрузите файл или введите текст вручную.")
    else:
        # Вызываем функцию process_files, которая объединяет текст из файлов и из ручного ввода,
        # затем разбивает текст на части, векторизует их и отправляет результат на Webhook.
        result = process_files(uploaded_files, manual_text, model, webhook_url)
        if result:
            st.success("Файлы успешно обработаны и данные отправлены на Webhook!")
        else:
            st.error("Произошла ошибка при обработке данных.")