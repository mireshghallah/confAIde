#load stories
#remove the what should ...


import os
import time
import openai
from types import SimpleNamespace
res_dict=['_generic','_revealing','_vague']
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


    return output


suffix='_nointent'

scenarios_file = open(f'/home/niloofar/privacy-project/data_gen/tier_3_prompts_subset{suffix}.txt','r')

scenarios_outs_qs = open(f'/home/niloofar/privacy-project/data_gen/tier_3_prompts_responses{suffix}.txt','w')
scenarios_outs_free = open(f'/home/niloofar/privacy-project/data_gen/tier_3_prompts_processed{suffix}.txt','w')


#g_truth = open('','r')


#questions = ['What is the sensitive information in this scenario?', f'Who is privy to the sensitive information {} in this scenario?','Who knows the sensitive information when the scenario ends?' ]

template_file = open(f'/home/niloofar/privacy-project/ToM_scenarios/gen_template_responses.txt','r')

template = template_file.read()

template_file.close()

def get_responses(scenario, person):
    scenario = scenario.strip()
    prompt = template + '\n'+ f'Now, you generate the set of three responses (generic, revealing and vague) for the following: \n Scenario:{scenario}. {person} said: \n'
    # print("here")
    # print('prompt is:',prompt)
    responses = prompt_model(prompt=prompt)
    print("RESPONSE")
    print(responses)
    _,_,responses = responses.partition('Response 1 (generic):')
    responses = responses.strip().split('\n')
    update_responses = []
    for response in responses:
        if response != '':
            update_responses.append(response)
    responses = update_responses[::2]

    print("responses are:")
    print(responses)
    #responses = [':'.join(response.strip().split(':')[1:]) for response in responses]
    return f'Scenario:{scenario}. {person} said:\n',responses



scenario_string = ''
full_s_responses = ''
full_s_free = ''
full_s = ''
people_list = []
scenario_flag = False
people_flag = False
for line in scenarios_file:





    if line.startswith('<BEGIN>') :
        print("begin")
        full_s += line
        full_s_free+= line
        scenario_flag = True
    elif line.startswith('<END>'):
        scenario_flag=False
        print('end')
        out= line.strip()
        list_of_words = out.split()
        before_keyword, keyword, after_keyword = out.partition('Questionee')
        people_s = after_keyword
        out+= f'<{people_s}>'
        ## get responses
        teller=after_keyword.split()[0]
                                
        full_scenario, responses =get_responses(scenario_string,teller) #template

        for i,response in enumerate(responses):
            scenarios_outs_qs.write(full_s+full_scenario+response+'\n'+f'<response{res_dict[i]}>'+out+'\n')
            scenarios_outs_qs.flush()
        ##
        scenarios_outs_free.write(full_s_free+out+'\n')
        scenarios_outs_free.flush()

        full_s_free = ''
        full_s = ''
        people_list = []
        scenario_string =''
        people_flag = False


    # elif line.strip().lower().startswith('people involved'):
    #     print('people start')
    #     people_flag= True
    #     scenario_flag=False

        
    elif scenario_flag:
        print('scenario')
        full_s_free += line
        if 'what should' not in line.lower():
            scenario_string += line
        else:
            scenario_string += '.'.join(line.split('.') [:-1])
    elif people_flag:
        print("people")
        person = line.strip().split(':')[-1]
        people_list.append(person)


    

