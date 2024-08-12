import json
import time

import google.generativeai as genai
import PIL.Image
import os

import inflect

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

p = inflect.engine()


class GeminiPrompter:
    def __init__(self, puzzle, n_examples, model="gemini-1.5-flash", verbose=False):
        self._images = [puzzle]
        self._model = model
        self._verbose = verbose
        self._n_examples = n_examples

        with open(f"{os.path.dirname(__file__)}/prompts.json", "r") as file:
            self._prompts = json.load(file)

        if self._n_examples > 0:
            examples = self._prompts["few_shot_examples"][str(self._n_examples)]
            examples_desc = "".join(
                f"\n- Image {i + 1}: {example["description"]}" for i, example in enumerate(examples))
            preprompt_template = self._prompts["prompt_templates"]["few_shot"]["preprompt"]
            if self._n_examples == 1:
                preprompt_template = preprompt_template["1"]
                self._preprompt = preprompt_template.format(*[examples_desc])
            else:
                preprompt_template = preprompt_template["2+"]
                cardinal_num = p.number_to_words(self._n_examples)
                self._preprompt = preprompt_template.format(*[cardinal_num, examples_desc])
            examples_images = [f"{os.path.dirname(__file__)}/../results/benchmark/final_v3/{example["image"]}"
                               for example in examples]
            self._images = examples_images + [puzzle]
        else:
            self._preprompt = self._prompts["prompt_templates"]["zero_shot"]["preprompt"]

    def send_prompt(self, text, image_paths=None, max_retries=10, timeout=10, max_tokens=300):
        model = genai.GenerativeModel(model_name=self._model)

        def _send():
            if image_paths is not None:
                image = PIL.Image.open(image_paths[0], "r")
                return model.generate_content([text, image],
                                              generation_config=genai.GenerationConfig(max_output_tokens=max_tokens))
            return model.generate_content(text, generation_config=genai.GenerationConfig(max_output_tokens=max_tokens))

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

    def score_statement(self, statement):
        prompt_template = self._prompts["prompt_templates"]["zero_shot"]["score_statement"]
        text_value = prompt_template.format(*[self._preprompt, statement])
        response = self.send_prompt(text_value, image_paths=self._images, max_tokens=1)
        value = response.text

        prompt_template = self._prompts["prompt_templates"]["zero_shot"]["gemini"]["score_statement"]
        text_prob = prompt_template.format(*[self._preprompt, value.lower(), statement])
        response = self.send_prompt(text_prob, image_paths=self._images, max_tokens=3)
        prob = float(response.text)

        return value, prob

    def generate_premise_from_hypothesis(self, hypothesis):
        prompt_template = self._prompts["prompt_templates"]["zero_shot"]["generate_premises"]
        text = prompt_template.format(*[self._preprompt, hypothesis])
        response = self.send_prompt(text, image_paths=self._images)
        premises = response.text
        return premises

    def generate_negated_statement(self, statement):
        prompt_template = self._prompts["prompt_templates"]["generate_negation"]
        text = prompt_template.format(*[statement])
        response = self.send_prompt(text)
        return response.text

    def score_rule(self, premises, hypothesis):
        premises_text = "".join([f"\n- {premise}" for premise in premises if premise != ""])
        prompt_template = self._prompts["prompt_templates"]["zero_shot"]["gemini"]["score_rule"]
        text = prompt_template.format(*[self._preprompt, premises_text, hypothesis])
        response = self.send_prompt(text, image_paths=self._images, max_tokens=3)
        # print(response.text)
        return response.text

