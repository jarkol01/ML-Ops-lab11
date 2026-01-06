from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np
from mangum import Mangum

app = FastAPI()

handler = Mangum(app)

tokenizer = Tokenizer.from_file("model/tokenizer.json")
ort_session = ort.InferenceSession("model/embedding.onnx")
ort_classifier = ort.InferenceSession("model/classifier.onnx")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str

SENTIMENT_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # tokenize input
    encoded = tokenizer.encode(request.text)

    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    # run embedding inference
    embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    embeddings = ort_session.run(None, embedding_inputs)[0]

    # run classifier inference
    classifier_input_name = ort_classifier.get_inputs()[0].name
    classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
    prediction = ort_classifier.run(None, classifier_inputs)[0]

    label = SENTIMENT_MAP.get(prediction[0], "unknown") # return this label as response

    return PredictResponse(
        label=label,
    )