import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={"": 0},torch_dtype=torch.float16
)
