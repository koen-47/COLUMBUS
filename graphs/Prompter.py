import os
import time
import requests
import base64

import openai

openai.key = os.getenv("OPENAI_API_KEY")


class Prompter:
    def __init__(self, model="gpt-4o-mini"):
        self._model = model

    def send_image_text_prompt(self, text, image_path=None, max_retries=30, timeout=30, temperature=0.7):
        with open(image_path, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()
