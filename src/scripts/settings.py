class Settings:
    bucket_name: str = 'mlops-lab11-models-jkoldun'
    local_dir: str = "model"
    classifier_joblib_path: str = "model/classifier.joblib"
    onnx_classifier_path: str = "model/classifier.onnx"
    embedding_dim: int = 384
    sentence_transformer_dir: str = "model/sentence_transformer.model"
    onnx_embedding_model_path: str = "model/embedding.onnx"
    onnx_tokenizer_path: str = "model/tokenizer.json"
