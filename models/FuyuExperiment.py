from transformers import FuyuProcessor, FuyuForCausalLM, BitsAndBytesConfig
import torch

from .ModelExperiment import ModelExperiment


class FuyuExperiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = "Fuyu-8b"
        self.prompt = "Which word/phrase best describes this image?\n" \
                      "(A) {}\n" \
                      "(B) {}\n" \
                      "(C) {}\n" \
                      "(D) {}\n"
        self._load_model()

    def _load_model(self):
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        self.model = FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0},
            torch_dtype=torch.float16
        )