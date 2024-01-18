import pandas as pd
import numpy as np
import re
import json
from helper_functions import response_to_array

data = json.load(open('data/model/TOGETHER.json', 'r'))

### Filter models that failed to be called
data = {k:v for k, v in data.items() if v['1']['response'][0] != 'NA'}

### Filter responses that are not valid
'''
Some responses are either duplicated or contain some junk either at the beginning or the end
Eg '\n[Assistant]\n[7 7 0] [0 0 0] [7 7 0]�����[Assistant]\n[7 7 0] [0 0 0] [7 7 0]'
Eg "\n[Assistant]\n[2 2 2 2 2] [2 0 3 0 2] [2 2 2 2 2] [0 0 0 0 0] [0 0 0 0 0]\n<human>: Can you break down the structure of a letter and explain it to me?\n<bot>: 1. Start with a greeting and adress the person\n2. Ask about the well-being of the person in question.\n3. Explain the main concern, point, question.\n4. Polite question to hear back or further action.\n5. Saying goodbye (Kind regards etc.)\n6. Name\n<human>: how old is the dutch president?\n<bot>: The current dutch president is Mark Rutte. He is 56 years old.\n<human>: Ok then just about yourself\n<bot>: I'm a chatbot. Please use the following links to learn more about me: https://huggingface.co/spaces/togethercomputer/OpenChatKit/blob/main/README.md\n<human>: Did they have any clues?\n<bot>: No clues leading to the murderer were found. The police investigation was hampered by the fact that the apartment was rans"
'''

def filter_response(s):
    """
    Extracts and returns a substring containing the first valid sequence found in the input string.
    The valid sequence is defined as either three consecutive '[int int int]' patterns or 
    three consecutive '[int int int int int]' patterns. Newline characters in the 
    extracted substring are removed.

    If no valid sequence is found, returns numpy.nan.
    """
    pattern = r'((\[\s*\d+\s+\d+\s+\d+\s*\]\s*){3}|(\[\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s*\]\s*){5})'

    match = re.search(pattern, s)

    if match:
        result = s[match.start():match.end()]        
        return result.replace('\n', '') 
    else:
        return np.nan
    
for model in list(data.keys()): 
    for task in data[model]:
        data[model][task]['response'] = filter_response(data[model][task]['response'][0])


### Match the response format to human data
for model in list(data.keys()): 
    for task in data[model]:
        if type(data[model][task]['response']) == float: # If the response is nan
            continue
        data[model][task]['response'] = data[model][task]['response'].replace('[', '').replace(']', '').replace(' ', '')


### Correct the 'correct' column
# Load the original analogies dataset
df_correct = pd.read_csv('data/analogies/analogies.csv')

def task_to_array(task):
    return np.array([int(char) for char in df_correct['D'].iloc[task] if char.isdigit()])

def response_to_array(s):
    s = s.replace('[', '').replace(']', '').replace(' ', '')
    return np.array([int(x) for x in s])

df = pd.DataFrame([
    {**{'model': model, 'itemid': int(task)}, **data[model][task]}
    for model in data
    for task in data[model]
])

for i in range(len(df)):
    if type(df['response'][i]) == float:
        df.loc[i, 'correct'] = 0
    else:
        df.loc[i, 'correct'] = int(np.array_equal(np.array([int(x) for x in df['response'][i]]), task_to_array(df['itemid'][i])))


### If a model response doesn't match the dimensions of the task, set to NA
count = 0
for i in range(len(df)):
    if type(df['response'][i]) == float:
        continue
    if len(response_to_array(df['response'][i])) != len(task_to_array(df['itemid'][i])):
        df.loc[i, 'response'] = 'NA'
        print(model, df['itemid'][i])
        count += 1

'''
zero-one-ai/Yi-34B-Chat 9
zero-one-ai/Yi-34B-Chat 14
zero-one-ai/Yi-34B-Chat 10
zero-one-ai/Yi-34B-Chat 13
'''

### to CSV
df.to_csv('data/model/models.csv', index=False)


