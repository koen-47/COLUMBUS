import json

import torch
from tqdm import tqdm
from PIL import Image

from .PromptManager import PromptManager


class ModelExperiment:
    def __init__(self):
        self.questions = PromptManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.name = None
        self.prompt = None
        
    def run_on_benchmark(self, save_dir):
        for puzzle in tqdm(self.questions.compounds, desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"])
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device, dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=128)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{self.name.lower()}_compounds.json", "w+") as file:
            json.dump(self.questions.compounds, file, indent=3)

        for puzzle in tqdm(self.questions.phrases, desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"])
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(device=self.device, dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=128)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            puzzle["output"] = generated_text

        with open(f"{save_dir}/{self.name.lower()}_phrases.json", "w+") as file:
            json.dump(self.questions.compounds, file, indent=3)
