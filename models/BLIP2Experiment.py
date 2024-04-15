import json
import os

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
from tqdm import tqdm

from .ModelExperiment import ModelExperiment


class BLIP2Experiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = "BLIP-2"
        self.prompt = "Question: which word/phrase best describes this image?\n" \
                      "(A) {}\n" \
                      "(B) {}\n" \
                      "(C) {}\n" \
                      "(D) {}\n" \
                      "Answer:"
        self._load_model()

    def _load_model(self):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-6.7b", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0},
            torch_dtype=torch.float16
        )


