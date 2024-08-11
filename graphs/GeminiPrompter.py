import time

import google.generativeai as genai
import PIL.Image
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class GeminiPrompter:
    def __init__(self, model="gemini-1.5-flash", verbose=False):
        self._model = model
        self._verbose = verbose

    def send_prompt(self, text, image_paths=None, max_retries=10, timeout=10, max_tokens=300):
        img = PIL.Image.open(image_paths[0], "r")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")

        def _send():
            response = model.generate_content([text, img], generation_config=genai.GenerationConfig(max_output_tokens=max_tokens))
            return response

        try:
            return _send()
        except:
            retries = 1
            while retries <= max_retries:
                try:
                    return _send()
                except:
                    if self._verbose:
                        print(f"Error (#{retries}). Resending prompt...")
                    retries += 1
                    time.sleep(timeout)
