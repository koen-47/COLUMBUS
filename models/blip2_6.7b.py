import json
import os

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch
from tqdm import tqdm

from models.PromptManager import PromptManager

prompter = PromptManager()
compound_image_questions = prompter.compound_image_question_pairs
phrase_image_questions = prompter.phrase_image_question_pairs

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-6.7b", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0},
#     torch_dtype=torch.float16
# )

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", device_map={"": 0}, torch_dtype=torch.float16
)

for puzzle in tqdm(compound_image_questions, desc="Prompting BLIP-2 (compounds)"):
    image = Image.open(puzzle["image"])
    question = puzzle["question"]
    inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    puzzle["output"] = generated_text

with open(f"{os.path.dirname(__file__)}/../results/experiments/blip2_6.7b_compounds.json", "w+") as file:
    json.dump(compound_image_questions, file, indent=3)


for puzzle in tqdm(phrase_image_questions, desc="Prompting BLIP-2 (phrases)"):
    image = Image.open(puzzle["image"])
    question = puzzle["question"]
    inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    puzzle["output"] = generated_text

with open(f"{os.path.dirname(__file__)}/../results/experiments/blip2_6.7b_phrases.json", "w+") as file:
    json.dump(phrase_image_questions, file, indent=3)

# print(f"\nImage: {os.path.basename(puzzle['image'])}")
# print(question.split("\n")[:-1])
# print(f"Output: {generated_text}")
# print(f"Correct: {puzzle['correct']}")
