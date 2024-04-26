from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch

from scripts.models.ModelExperiment import ModelExperiment


class BLIP2Experiment(ModelExperiment):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.name = f"BLIP-2 {size}"
        self.prompt = "Question: which word/phrase best describes this image?\n" \
                      "(A) {}\n" \
                      "(B) {}\n" \
                      "(C) {}\n" \
                      "(D) {}\n" \
                      "Answer:"
        self._load_model()

    def _load_model(self):
        self.processor = Blip2Processor.from_pretrained(f"Salesforce/blip2-opt-{self.size}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            f"Salesforce/blip2-opt-{self.size}", quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0}, torch_dtype=torch.float16
        )


