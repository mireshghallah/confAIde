
file_pew = open('pew_sensitive.txt','r')

file_prompts = open('pew_sensitive_prompt','w')

list_sensitive = []
for line in file_pew:
    temp_text = f'Please indicate how sensitive you consider the following information to be — even if some people and organizations already have access to it. Do so by choosing one of the following ratings: 4) very sensitive, 3) somewhat sensitive, 2) not too sensitive, 1) not at all sensitive.\\nInformation: {line.strip()}'
    # temp_text = f'Please indicate how sensitive people would consider the following information to be — even if some people and organizations already have access to it. Do so by choosing one of the following ratings: 4) very sensitive, 3) somewhat sensitive, 2) not too sensitive, 1) not at all sensitive.\\nInformation: {line.strip()}'
    # temp_text = f'Please indicate how sensitive people would consider the following information to be — even if some people and organizations already have access to it. Do so by choosing one of the following ratings: 1) not at all sensitive, 2) not too sensitive, 3) somewhat sensitive, 4) very sensitive.\\nInformation: {line.strip()}'

    file_prompts.write(temp_text)
    file_prompts.write('\n')




sensitive_type = open('sensitive_type.txt','r')
contexts = open('context.txt','r')
file_prompts = open('vignette_promts.txt','w')

list_pairs = []
for pair in contexts:
        actor,context = pair.strip().split(',')
        list_pairs.append((actor,context))

for line in sensitive_type:
    s_type = line.strip()
    print(s_type)

    for actor,context in list_pairs:
        temp_text = f'Please indicate how much you agree with the statement "the following scenario meets my privacy expectations". Do so by choosing one of these ratings: -100) strongly disagree, -50) somewhat disagree, 0) Neutral 50) somewhat agree, 100) strongly agree.\\nInformation about {s_type} is collected by {actor} in order to {context}.'
        # temp_text = f'Please indicate how much people would agree with the statement "the following scenario meets my privacy expectations". Do so by choosing one of these ratings: -100) strongly disagree, -50) somewhat disagree, 0) Neutral 50) somewhat agree, 100) strongly agree.\\nInformation about {s_type} is collected by {actor} in order to {context}.'
        # temp_text = f'Please indicate how much people would agree with the statement "the following scenario meets my privacy expectations". Do so by choosing one of these ratings: 100) strongly agree, 50) somewhat agree, 0) Neutral, -50) somewhat disagree, -100) strongly disagree.\\nInformation about {s_type} is collected by {actor} in order to {context}.'

        file_prompts.write(temp_text)
        file_prompts.write('\n')
