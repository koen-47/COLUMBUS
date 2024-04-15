import json
import os

from PIL import Image
from transformers import FuyuProcessor, FuyuForCausalLM
import torch
from tqdm import tqdm

from models.PromptManager import PromptManager

prompter = PromptManager()
compound_image_questions = prompter.compound_image_question_pairs
phrase_image_questions = prompter.phrase_image_question_pairs

print(len(compound_image_questions), len(phrase_image_questions))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

for puzzle in tqdm(compound_image_questions, desc="Prompting BLIP-2 (compounds)"):
    image = Image.open(puzzle["image"])
    question = puzzle["question"]
    inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    puzzle["output"] = generated_text

with open("./results/experiments/blip2_compounds.json", "w+") as file:
    json.dump(compound_image_questions, file, indent=3)


for puzzle in tqdm(phrase_image_questions, desc="Prompting BLIP-2 (phrases)"):
    image = Image.open(puzzle["image"])
    question = puzzle["question"]
    inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    puzzle["output"] = generated_text

with open("./results/experiments/blip2_phrases.json", "w+") as file:
    json.dump(phrase_image_questions, file, indent=3)
