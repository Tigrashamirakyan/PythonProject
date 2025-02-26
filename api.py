from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from process import split_text, vectorize_chunks, split_json_if_large, send_to_webhook
import json

app = FastAPI()


class VectorizationRequest(BaseModel):
    input_data: str
    model: str
    webhook_url: str


@app.post("/vectorize")
def vectorize(request: VectorizationRequest):
    if not request.input_data:
        raise HTTPException(status_code=400, detail="Нет входных данных")
    full_text = request.input_data
    chunks = split_text(full_text)
    vectors = vectorize_chunks(chunks, request.model)
    json_data = {
        "status": "success",
        "model": request.model,
        "original_text": full_text,
        "chunks": [
            {"chunk_id": i + 1, "text": chunk, "vector": vector}
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
    }
    # Для API используем размер 1 МБ для разделения (можно изменить по необходимости)
    max_size_bytes = 1 * 1024 * 1024
    files_data = split_json_if_large(json_data, max_size_bytes)
    results = []
    for data_part in files_data:
        status_code, response_text = send_to_webhook(request.webhook_url, data_part)
        results.append({"status_code": status_code, "response": response_text})
    return {"message": "Данные отправлены на Webhook", "results": results}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
