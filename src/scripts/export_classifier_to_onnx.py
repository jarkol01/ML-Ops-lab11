import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import save_onnx_model

from settings import Settings


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.classifier_joblib_path}...")
    classifier = joblib.load(settings.classifier_joblib_path)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(classifier, initial_types=initial_type)

    print(f"Saving ONNX model to {settings.onnx_classifier_path}...")
    save_onnx_model(onnx_model, settings.onnx_classifier_path)


if __name__ == "__main__":
    settings = Settings()
    export_classifier_to_onnx(settings)