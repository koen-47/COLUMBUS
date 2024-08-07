import base64
import os

import requests


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


image1_base64 = encode_image("../results/benchmark/final_v3/aftereffect.png")
image2_base64 = encode_image("../results/benchmark/final_v3/bearing_the_cross.png")
image3_base64 = encode_image("../results/benchmark/final_v3/back_the_wrong_horse_icon.png")
image4_base64 = encode_image("../results/benchmark/final_v3/down_on_one's_luck_icon.png")

payload = {
    "model": "gpt-4o",
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Refer to the images below:\n"
                        "Image 1: Describe what you see.\n"
                        "Image 2: Describe what you see.\n"
                        "Image 3: Describe what you see.\n"
                        "Image 4: Describe what you see.\n"

            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image1_base64}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image2_base64}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image3_base64}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image4_base64}"
                }
            }
        ]
    }],
    "max_tokens": 150,
    "logprobs": True
}

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv("OPENAI_API_KEY")}",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])
