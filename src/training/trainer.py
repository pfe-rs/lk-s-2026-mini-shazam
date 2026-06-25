class TrainingPipeline:
    def train_cnn(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")

    def export_onnx(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")

    def quantize_onnx(self) -> None:
        raise NotImplementedError("TrainingPipeline not yet implemented")
