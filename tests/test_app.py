from fastapi.testclient import TestClient
from src.sentiment_app.app import app


client = TestClient(app)


def test_inference():
    """
    Test that:
    - inference should work for a few sample strings
    - output should be a valid JSON response
    """
    sample_texts = [
        "I love this product!",
        "I have nothing to say.",
        "I hate this service.",
    ]

    expected_labels = ["positive", "neutral", "negative"]

    for index, text in enumerate(sample_texts):
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200

        result = response.json()
        assert "label" in result
        assert result["label"] == expected_labels[index]


def test_input_validation():
    """
    Test that:
    - input text should be a non-empty string
    - valid strings should be accepted
    """
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

    response = client.post("/predict", json={"text": "This is a valid text"})
    assert response.status_code == 200


def test_validation_error_response():
    """
    Test that:
    - for an invalid input, validation should return a valid JSON with error explanation
    """
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

    error_data = response.json()
    assert isinstance(error_data, dict)

    assert "detail" in error_data
    assert len(error_data["detail"]) > 0
    assert "msg" in error_data["detail"][0]
    assert error_data["detail"][0]["msg"] == "Value error, String should have at least 1 character"


def test_model_loading():
    """
    Test that:
    - model should be loaded from provided files without errors
    """
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("model/tokenizer.json")
    assert tokenizer is not None

    ort_session = ort.InferenceSession("model/embedding.onnx")
    assert ort_session is not None

    ort_classifier = ort.InferenceSession("model/classifier.onnx")
    assert ort_classifier is not None

