# import requests
# from PIL import Image
#
# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
#
# model_id = "llava-hf/llava-1.5-13b-hf"
#
# prompt = "USER: <image>\nWhich word/phrase best describes this image?\n" \
#                       "(A) birthwort\n" \
#                       "(B) birthroot\n" \
#                       "(C) afterbirth\n" \
#                       "(D) birthday\n" \
#                       "ASSISTANT:"
#
# image = Image.open("../results/compounds/saved/afterbirth.png").convert("RGB")
#
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     # use_flash_attention_2=True,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     quantization_config=BitsAndBytesConfig(load_in_8bit=True)
# )
#
# processor = AutoProcessor.from_pretrained(model_id)
#
# # raw_image = Image.open(requests.get(image, stream=True).raw)
# inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
#
# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))

import os

from LlavaExperiment import LlavaExperiment

llava = LlavaExperiment()
llava.run_on_benchmark(f"{os.path.dirname(__file__)}/../results/experiments")

