import os
import time
import openai
from types import SimpleNamespace

with open("oaik.txt", "r") as f:
    oaik = f.read()
os.environ['OPENAI_API_KEY'] = oaik

class GPTAgent():
    def __init__(self, kwargs: dict):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)

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

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)

        return output

    def parse_basic_text(self, response):
        output = response['choices'][0].message.content.strip()

        return output

class GPTAgentComplete():
    #     r = openai.Completion.create(prompt=f"{p}", **kwargs)
    #     return p + r['choices'][0].text
    def __init__(self, kwargs: dict):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)

    def generate(self, prompt):
        while True:
            try:
                completion = openai.Completion.create(
                    model=self.args.model,
                    prompt=f"{prompt}"
                )
                break
            except (openai.error.APIError, openai.error.RateLimitError) as e: 
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)

        return output

    def parse_basic_text(self, response):
        output = response['choices'][0].text.strip()

        return output



def prompt_model(suffix="", prompt="",prefix='j', model = 'gpt-4-0613'):
    #model = 'gpt-4-0613'
    if 'gpt-4' in model or 'turbo' in model:
        gpt = GPTAgent({'model': model, 'temperature': 0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
    else:
        gpt = GPTAgentComplete({'model': model, 'temperature': 0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})


    if prompt == "":
        with open(prefix+str(suffix)+".txt", "r") as f:
            prompt = f.read()

    output = gpt.interact(prompt)
    print(prompt)
    print(output)

    return output

    # with open(prefix + str(suffix) + "_output.txt", "a") as f:
    #     f.write(model + "\n")
    #     f.write(prompt + "\n")
    #     f.write(output + "\n\n")
    #     f.write("=============================================================================\n\n")

    # return output



file_name = 'tier_2_promts.txt'
tier_2 = open(file_name,'r')

# tier_3 = open('tier_3_prompts.txt','w')
tier_2b = open('tier_2-b_prompts.txt','w')


for i,prompt in enumerate(tier_2):
    prompt = prompt.strip()
    prompt_head = prompt.split('\\n')[0].strip()

    prompt = prompt.split('\\n')[-1].strip().replace('.','')

    new_prompt = f'Write a four sentence story in 2nd person narrative where "{prompt}". '

    output = prompt_model(prompt=new_prompt)

    # tier_3.write(f'{prompt_head}\\n s{output}')
    # tier_3.write('\n')

    prompt = prompt.split('\\n')[-1].strip().replace('.','')
    prompt = prompt.replace('Your','the main character\'s')
    new_prompt = f'Write a four sentence story in 3rd person narrative where "{prompt}". '

    output = prompt_model(prompt=new_prompt)

    tier_2b.write(f'{prompt_head}\\n {output}')
    tier_2b.write('\n')


