import openai

from graphs.Prompter import Prompter

prompter = Prompter()

text = "Describe what you see in this image."
image_path = "./results/benchmark/final_v3/aftereffect.png"
response = prompter.send_image_text_prompt(text, image_path)
print(response)
