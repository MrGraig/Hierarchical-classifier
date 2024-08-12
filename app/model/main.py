from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import re
import numpy as np
import uvicorn



model_dir = "model"
model_path = os.path.join(model_dir, 'model.pkl')
encoder_path = os.path.join(model_dir, 'encoder.pkl')
vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    word2vec_model = pickle.load(vectorizer_file)


# Функция предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'\W+', ' ', text)  
    text = text.strip()
    return text

# Функуия преобразования текста в вектор
def text_to_vector(text, word2vec_model):
    words = preprocess_text(text).split()
    words_in_vocab = [word for word in words if word in word2vec_model.wv]
    if len(words_in_vocab) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean([word2vec_model.wv[word] for word in words_in_vocab], axis=0)



app = FastAPI()

class TextInput(BaseModel):
    texts: list[str]

@app.get("/")
def read_root():
    return {"message": "Service is running."}

@app.post("/predict")
def predict(input_data: TextInput):
    try:
        # Преобразуем каждый текст в вектор
        X_transformed = np.array([text_to_vector(text, word2vec_model) for text in input_data.texts])
        
        y_pred = model.predict(X_transformed)
        
        # Преобразуем числовые предсказания обратно в текстовые метки
        y_pred_classes = label_encoder.inverse_transform(y_pred)
        
        return {"predictions": y_pred_classes.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)