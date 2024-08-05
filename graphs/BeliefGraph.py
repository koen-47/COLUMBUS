from .Prompter import Prompter


class RebusPuzzleBeliefGraph:
    def __init__(self, image_path, option):
        self._image_path = image_path
        self._hypothesis = f"The word/phrase conveyed in this image is {option}."
        self._prompter = Prompter()

    def generate(self):
        value, prob = self._get_statement_confidence(self._hypothesis)
        print("Statement:", self._hypothesis)
        print("Value:", value)
        print("Probability:", prob)

    def _extend_graph(self, statement, depth, max_depth, graph):
        pass

    def _get_statement_confidence(self, statement):
        text = f"{statement}: true or false?"
        response = self._prompter.send_image_text_prompt(text, image_path=self._image_path, max_tokens=1)
        content = response["choices"][0]["logprobs"]["content"][0]
        prob = 10 ** content["logprob"]
        value = content["token"]
        return value, prob
