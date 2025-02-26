# Используем официальный образ Python (версия 3.10-slim, можно изменить на нужную)
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей и устанавливаем зависимости
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в контейнер
COPY . /app

# Открываем порт 8501, который по умолчанию используется Streamlit
EXPOSE 8501

# Команда запускает Streamlit, слушая на 0.0.0.0 и порту 8501.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
