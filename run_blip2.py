from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch

from models.PromptManager import PromptManager

prompter = PromptManager()
compound_image_questions = prompter.compound_image_question_pairs

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0},
    torch_dtype=torch.float16
)

for puzzle in compound_image_questions:
    image = puzzle["image"]
    question = puzzle["question"]
    inputs = processor(images=image, text=question, return_tensors="pt").to(device=device, dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_length=128)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"Image: {image}")
    print(question)
    print(f"Correct: {puzzle['correct']}")
