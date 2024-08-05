import os
import time
import requests
import base64

import openai

openai.key = os.getenv("OPENAI_API_KEY")


class Prompter:
    def __init__(self, model="gpt-4o-mini", verbose=False):
        self._model = model
        self._verbose = verbose

    def send_prompt(self, text, image_path=None, max_retries=10, timeout=10, max_tokens=300):
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
                        }

                    ]
                }
            ],
            "max_tokens": max_tokens,
            "logprobs": True
        }

        if image_path is not None:
            with open(image_path, "rb") as image_file:
                image = base64.b64encode(image_file.read()).decode("utf-8")

            payload["messages"][0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    }
                }
            )

        def _send():
            response_ = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            return response_.json()

        response = None
        try:
            response = _send()
        except:
            retries = 1
            while retries <= max_retries:
                try:
                    response = _send()
                    break
                except:
                    if self._verbose:
                        print(f"Error (#{retries}). Resending prompt...")
                    retries += 1
                    time.sleep(timeout)

        if "error" in response:
            retries = 1
            while retries <= max_retries and "error" in response:
                if self._verbose:
                    print(f"Error (#{retries}). Resending prompt...")
                retries += 1
                time.sleep(timeout)
                response = _send()
            return response
        else:
            return response
