import json
import os

from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from models.ModelExperiment import ModelExperiment
from data.Benchmark import Benchmark


class LlavaExperiment(ModelExperiment):
    def __init__(self):
        super().__init__()
        self.name = "Llava-1.5-13b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models_dir = f"{os.path.dirname(__file__)}/downloads"
        self.prompt = "USER: <image>\n" \
                      "Which word/phrase is conveyed in this image from the following options (either A, B, C, or D)? " \
                      "(A) {} (B) {} (C) {} (D) {}\n" \
                      "ASSISTANT:"
        self._load_model()

    def _load_model(self):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
            cache_dir=self._models_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-13b-hf",
            cache_dir=self._models_dir
        )

    def run_on_benchmark(self, save_dir):
        benchmark = Benchmark()
        compounds, phrases = benchmark.get_puzzles()

        print(f"Starting experiment: {self.name}")
        print(f"Number of compounds: {len(compounds)}")
        print(f"Number of phrases: {len(phrases)}")
        print(f"Results save directory: {save_dir}")

        for puzzle in tqdm(compounds[:1], desc=f"Prompting {self.name} (compounds)"):
            image = Image.open(puzzle["image"]).convert("RGB")
            options = puzzle["options"]
            prompt = self.prompt.format(*options.values())
            puzzle["prompt"] = prompt
            inputs = self.processor(prompt, image, return_tensors='pt').to(device=self.device,
                                                                           dtype=torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)
            puzzle["output"] = generated_text
            print(generated_text)

        # with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_compounds.json", "w+") as file:
        #     json.dump(compounds, file, indent=3)
        #
        # for puzzle in tqdm(phrases, desc=f"Prompting {self.name} (phrases)"):
        #     image = Image.open(puzzle["image"]).convert("RGB")
        #     options = puzzle["options"]
        #     prompt = self.prompt.format(*options.values())
        #     puzzle["prompt"] = prompt
        #     inputs = self.processor(prompt, image, return_tensors='pt').to(device=self.device,
        #                                                                    dtype=torch.float16)
        #     output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        #     generated_text = self.processor.decode(output[0][2:], skip_special_tokens=True)
        #     puzzle["output"] = generated_text
        #
        # with open(f"{save_dir}/{'_'.join(self.name.lower().split())}_phrases.json", "w+") as file:
        #     json.dump(phrases, file, indent=3)
