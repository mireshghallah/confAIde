import torch
from types import SimpleNamespace
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
import transformers


class HuggingFaceAgent():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_text(self, text):
        return text

    def postprocess_output(self, response):
        return response

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer(prompt, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=self.do_sample)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=self.do_sample)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class FlanT5Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + args.model)
        self.model = T5ForConditionalGeneration.from_pretrained("google/" + args.model, device_map="auto")

    def interact(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class FlanUL2Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
        self.do_sample = args.do_sample_for_local_models

class OPTAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model)
        self.model = AutoModelForCausalLM.from_pretrained("facebook/" + args.model, device_map="auto")

class DollyAgent():
    def __init__(self):
        self.generate_text = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    def interact(self, prompt):
        result = self.generate_text(prompt)
        return result[0]["generated_text"]

class KoalaAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/{}-HF".format(args.model))
        self.model = AutoModelForCausalLM.from_pretrained("TheBloke/{}-HF".format(args.model), device_map="auto", load_in_8bit=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token

    def preprocess_text(self, text):
        prompt = "BEGINNING OF CONVERSATION: USER: {} GPT:".format(text)
        return prompt

    def postprocess_output(self, response):
        if "GPT:" in response:
            response = response.split("GPT:")[-1].strip()
        elif "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip()
        elif "BEGINNING OF CONVERSATION: USER:" in response:
            response = "" # if there is not GPT: and only BEGINNING OF CONVERSATION: USER:, then the model did not generate a response

        return response

    def update_context(self, previous_context, response, new_input):
        updated_context = "{}{}</s>USER: {} GPT:".format(previous_context, response, new_input)
        return updated_context

class VicunaAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("eachadea/vicuna-7b-1.1")
        self.model = AutoModelForCausalLM.from_pretrained("eachadea/vicuna-7b-1.1", device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token

    def postprocess_output(self, response):
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip().removeprefix(". ")
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip().removeprefix(". ")
        elif response.startswith("This is a theory-of-mind test"):
            response = "" # if there is not Answer: and only the instruction, then the model did not generate a response

        return response

class FalconAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)

        model = "tiiuae/{}".format(args.model)

        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side='left')
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            max_length=800,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id

    def postprocess_output(self, response):
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip().removeprefix(". ")
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip().removeprefix(". ")
        elif response.startswith("This is a theory-of-mind test"):
            response = "" # if there is not Answer: and only the instruction, then the model did not generate a response

        return response

    def interact(self, text):
        sequences = self.pipeline(
            text,
            max_length=800,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]['generated_text']

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        sequences = self.pipeline(
            batch_prompts,
            max_length=800,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded_outputs = [seq[0]['generated_text'] for seq in sequences]
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class Llama2Agent(HuggingFaceAgent):
    def __init__(self, args):
        if type(args) is dict:
            args = SimpleNamespace(**args)
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/{}".format(args.model))
        if "70b" in args.model or "13b" in args.model:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/{}".format(args.model), device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/{}".format(args.model), device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer(prompt, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        # attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=128)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class UnifiedQAv2Agent(HuggingFaceAgent):
    def __init__(self, kwargs: dict):
        self.args = SimpleNamespace(**kwargs)
        super().__init__(self.args)
        self.tokenizer = T5Tokenizer.from_pretrained("allenai/{}".format(self.args.model))
        self.model = T5ForConditionalGeneration.from_pretrained("allenai/{}".format(self.args.model), device_map="auto")

    def interact(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids, max_new_tokens=128, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)
    
    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class GuanacoAgent(HuggingFaceAgent):
    def __init__(self, kwargs: dict):
        self.args = SimpleNamespace(**kwargs)
        super().__init__(self.args)
        # self.tokenizer = AutoTokenizer.from_pretrained("timdettmers/guanaco-33b-merged")
        # self.model = AutoModelForCausalLM.from_pretrained("timdettmers/guanaco-33b-merged", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/guanaco-13B-HF")
        self.model = AutoModelForCausalLM.from_pretrained("TheBloke/guanaco-13B-HF", device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess_text(self, text):
        formatted_prompt = (
            f"A chat between a curious human and an artificial intelligence assistant."
            f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            f"### Human: {text} ### Assistant:"
        )

        return formatted_prompt

    def postprocess_output(self, response):
        return response.split("### Assistant:")[1].strip()

    def interact(self, prompt):
        formatted_prompt = self.preprocess_text(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs=inputs.input_ids, max_new_tokens=128)
        _response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self.postprocess_output(_response)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses
    