import json

from tqdm import tqdm

from models.ModelExperiment import ModelExperiment

import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from models.PromptManager import PromptManager


class LlavaExperiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = f"Llava-1.5-13b"
        self.questions = PromptManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt = "USER: <image>\n" \
                      "Which word/phrase best describes this image?\n" \
                      "(A) {}\n" \
                      "(B) {}\n" \
                      "(C) {}\n" \
                      "(D) {}\n" \
                      "ASSISTANT:"
        self._load_model()

    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
            # use_flash_attention_2=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

    def run_on_benchmark(self, save_dir):
        for puzzle in tqdm(self.questions.compounds, desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            inputs = self.processor(prompt, image, return_tensors='pt').to(0, torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            output = self.processor.decode(output[0][2:], skip_special_tokens=True)
            puzzle["output"] = output

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds.json", "w+") as file:
            json.dump(self.questions.compounds, file, indent=3)

        for puzzle in tqdm(self.questions.phrases, desc=f"Prompting {self.name} (phrases)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device, dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=512)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases.json", "w+") as file:
            json.dump(self.questions.phrases, file, indent=3)
