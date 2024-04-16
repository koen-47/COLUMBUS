from transformers import FuyuProcessor, FuyuForCausalLM, BitsAndBytesConfig
from PIL import Image
import requests

# load model and processor
processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda:0")

text_prompt = "What do you see in this image?\n"
image = Image.open("../results/compounds/saved/afterbirth.png").convert("RGB")

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

generation_output = model.generate(**inputs, max_length=512)
generation_text = processor.batch_decode(generation_output, skip_special_tokens=True)
print(generation_text)

# import json
# import os
#
# from PIL import Image
# from transformers import FuyuProcessor, FuyuForCausalLM
# import torch
# from tqdm import tqdm
#
# from models.PromptManager import PromptManager
#
# prompter = PromptManager()
# compound_image_questions = prompter.compound_image_question_pairs
# phrase_image_questions = prompter.phrase_image_question_pairs
#
# print(len(compound_image_questions), len(phrase_image_questions))
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Device:", device)
#
# model_id = "adept/fuyu-8b"
# processor = FuyuProcessor.from_pretrained(model_id)
# model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")
#
# for puzzle in tqdm(compound_image_questions, desc="Prompting BLIP-2 (compounds)"):
#     image = Image.open(puzzle["image"])
#     question = puzzle["question"]
#     inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
#     generated_ids = model.generate(**inputs, max_length=128)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     puzzle["output"] = generated_text
#
# with open("./results/experiments/blip2_compounds.json", "w+") as file:
#     json.dump(compound_image_questions, file, indent=3)
#
#
# for puzzle in tqdm(phrase_image_questions, desc="Prompting BLIP-2 (phrases)"):
#     image = Image.open(puzzle["image"])
#     question = puzzle["question"]
#     inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
#     generated_ids = model.generate(**inputs, max_length=128)
#     generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     puzzle["output"] = generated_text
#
# with open("./results/experiments/blip2_phrases.json", "w+") as file:
#     json.dump(phrase_image_questions, file, indent=3)
