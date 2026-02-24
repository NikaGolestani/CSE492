import tensorflow as tf


class TFLiteFloat16Converter:

    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        self.model = None

    def load_model(self):
        print(f"[INFO] Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False
        )

    def convert(self):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        print("[INFO] Converting model to TFLite (float16)...")

        converter = tf.lite.TFLiteConverter.from_keras_model(
            self.model
        )

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        return tflite_model

    def save(self, tflite_model):
        print(f"[INFO] Saving TFLite model to: {self.output_path}")

        with open(self.output_path, "wb") as f:
            f.write(tflite_model)

        print("[INFO] Conversion complete.")

    def run(self):
        self.load_model()
        tflite_model = self.convert()
        self.save(tflite_model)


if __name__ == "__main__":
    MODEL_PATH = "saliency_x10_x3_final.keras"
    TFLITE_NAME = "saliency_float16.tflite"

    converter = TFLiteFloat16Converter(
        MODEL_PATH,
        TFLITE_NAME
    )

    converter.run()