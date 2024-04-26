import os

import requests
from PIL import Image
from transformers import FuyuProcessor, FuyuForCausalLM, BitsAndBytesConfig
import torch

from scripts.models.ModelExperiment import ModelExperiment


class FuyuExperiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = "Fuyu-8b"
        self.prompt = "Which word/phrase best describes this image?\n" \
                      "(A) {}\n" \
                      "(B) {}\n" \
                      "(C) {}\n" \
                      "(D) {}\n"
        # self._load_model()

    def _load_model(self):
        self.processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
        self.model = FuyuForCausalLM.from_pretrained(
            "adept/fuyu-8b",
            device_map={"": 0},
        )

    def run_on_benchmark(self, save_dir):
        image_path = f"{os.path.dirname(__file__)}/../data/images/compounds/afterbirth.png"
        print(image_path)

        image = Image.open(image_path).convert("RGB")


        # inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

        # autoregressively generate text
        # generation_output = self.model.generate(**inputs, max_new_tokens=7)
        # generation_text = self.processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
