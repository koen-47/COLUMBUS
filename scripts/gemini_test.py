from graphs.GeminiPrompter import GeminiPrompter

prompter = GeminiPrompter()

image = ["../results/benchmark/final_v3/a_bird_in_the_hand_icon.png"]
response = prompter.send_prompt("What do you see in this image?", image, max_tokens=1)
print(response)
