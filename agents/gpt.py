import os
import time
import openai
from types import SimpleNamespace

class GPT3BaseAgent():
    def __init__(self, kwargs: dict):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def _set_default_args(self):
        if not hasattr(self.args, 'engine'):
            self.args.engine = "text-davinci-003"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt):
        while True:
            try:
                completion = openai.Completion.create(
                    engine=self.args.engine,
                    prompt=prompt,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    stop=self.args.stop_tokens if hasattr(self.args, 'stop_tokens') else None,
                    logprobs=self.args.logprobs if hasattr(self.args, 'logprobs') else 0,
                    echo=self.args.echo if hasattr(self.args, 'echo') else False
                )
                break
            except (RuntimeError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.APIConnectionError) as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def parse_basic_text(self, response):
        output = response['choices'][0]['text'].strip()

        return output

    def parse_ordered_list(self, numbered_items):
        ordered_list = numbered_items.split("\n")
        output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]

        return output

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)

        return output

class GPT3QAAgent(GPT3BaseAgent):
    """
    A simple QA agent that uses GPT-3 to answer questions with predefined answer candidates.
    The answer candiates are ranked by their log probabilities.
    The default mode is for simple yes or no questions.
    """

    def __init__(self, prompt_header=None, max_tokens=6, top_p=1, stop_tokens=['.'], answer_candidates=[" Yes", " No"]):
        super().__init__({'temperature': 0, 'max_tokens': max_tokens, 'top_p': top_p, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'logprobs': 100, 'stop_tokens': stop_tokens})

        self.prompt_header = prompt_header if prompt_header is not None else "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. Answer in yes, no, or unknown.\n\n"

        self.answer_candidates = answer_candidates

    def vanilla_argmax(self, token_logprobs):
        answer_candidates_logp = {token: 0 for token in self.answer_candidates}
        for i, candidate in enumerate(answer_candidates_logp.keys()):
            # Check whether that answer candidate is in the top logprobs
            if candidate in token_logprobs[0].keys():
                answer_candidates_logp[candidate] = token_logprobs[0][candidate]
            else:
                answer_candidates_logp[candidate] = -float('inf')
        answer = max(answer_candidates_logp, key=answer_candidates_logp.get)

        return answer.strip(), answer_candidates_logp[answer]

    def get_answer(self, question):
        prompt = "{}{}\nA:".format(self.prompt_header, question)
        response = self.generate(prompt)
        logprobs = {
            "token_logprobs": response['choices'][0]['logprobs']['token_logprobs'],
            "top_logprobs": response['choices'][0]['logprobs']['top_logprobs'],
            "tokens": response['choices'][0]['logprobs']['tokens'],
        }

        answer, answer_logprob = self.vanilla_argmax(logprobs['top_logprobs'])

        return answer, answer_logprob

class GPT3MultipleChoiceQAAgent(GPT3BaseAgent):
    """
    A simple QA agent that uses GPT-3 to answer multiple-choice questions with predefined answer options.
    The default mode is for simple yes, no, unknown questions.
    """
    def __init__(self, prompt_header=None, prompt_footer=None, max_tokens=8, top_p=1, stop_tokens=[".", "\n"], options=["Yes", "No", "Unknown"]):
        super().__init__({'engine': "text-davinci-003",'temperature': 0, 'max_tokens': max_tokens, 'top_p': top_p, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'logprobs': 100, 'stop_tokens': stop_tokens})

        self.prompt_header = prompt_header if prompt_header is not None else "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer.\n\n"

        if prompt_footer is not None:
            self.prompt_footer = prompt_footer
        else:
            # iterate through options and create a prompt footer: (a) option1 (b) option2 (c) option3
            self.prompt_footer = ""
            for idx, candidate in enumerate(options):
                self.prompt_footer += "\n({}) {}".format(chr(97+idx), candidate)
            self.prompt_footer += "\n\nChoose an answer:"

        self.options = options

    def get_answer(self, question):
        prompt = "{}{}{}".format(self.prompt_header, question, self.prompt_footer)
        response = self.interact(prompt).lower()

        for alphabet in [chr(97+i) for i in range(len(self.options))]:
            if "(" + alphabet + ")" in response:
                return self.options[ord(alphabet)-97]

        # sort options by their length in descending order, to prevent short options from matching long options - e.g., "no" from matching "unknown"
        sorted_options = sorted(self.options, key=len, reverse=True)
        for option in sorted_options:
            if option.lower() in response:
                return option


class ConversationalGPTBaseAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-4-0613"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.args.model,
                    messages=[{"role": "user", "content": "{}".format(prompt)}]
                )
                break
            except (openai.error.APIError, openai.error.RateLimitError) as e: 
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def parse_basic_text(self, response):
        output = response['choices'][0].message.content.strip()

        return output