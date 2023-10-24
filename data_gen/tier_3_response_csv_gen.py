#load stories
#remove the what should ...


import os
import time
from types import SimpleNamespace
import csv


#suffix='_nointent'

response_out = open(f'/home/niloofar/privacy-project/benchmark/tier_3_scenarios_responses_human.csv','w')

scenarios_file_responses = open(f'/home/niloofar/privacy-project/benchmark/tier_3_scenarios_responses.txt','r')


#g_truth = open('','r')


#questions = ['What is the sensitive information in this scenario?', f'Who is privy to the sensitive information {} in this scenario?','Who knows the sensitive information when the scenario ends?' ]


writer = csv.writer(response_out)
writer.writerow(['scenario_idx', 'scenario', 'generic','revealing'])

scenario_string = ''
full_s_responses = ''
full_s_free = ''
full_s = ''
people_list = []
scenario_flag = False
scen_start = 'generic'
idx = 0
temp_l = []
curr_res  = ''
for line in scenarios_file_responses:


    if line.startswith('<BEGIN>')  and scen_start == 'generic':
        temp_l.append(idx)
        idx+=1
        
        print("begin generic for ", idx)
        
        scenario_flag = True
    elif line.startswith('\"') and scen_start == 'generic':
        print('adding generic for ', idx)
        scenario_flag=False
        curr_res = line.strip()
    elif line.startswith('\"') and scen_start == 'revealing':
        print('adding and writing revealing for ',idx,line.strip())
        scenario_flag=False
        temp_l.append(line.strip())
    
    elif line.startswith('<response_generic>'):
        scen_start = 'revealing'
        temp_l.append(scenario_string)
        temp_l.append(curr_res.strip())
        scenario_string = ''
        print('writing for generic scenario and ', idx, curr_res)

    elif line.startswith('<response_revealing>'):
        scen_start = 'vague'

    elif line.startswith('<response_vague>'):
        scen_start = 'generic'
        print("writing row for " ,idx)
        writer.writerow(temp_l)
        temp_l = []

        
    elif scenario_flag:
        print('scenario for', idx)
        full_s_free += line
        
        scenario_string += line
 
    

