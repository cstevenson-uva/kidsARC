import os
import requests
import json
import time

from helper_functions import task_to_base64, task_to_input, check_response

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.Client()

import together
together.api_key = os.environ["TOGETHER_API_KEY"]
from typing import Optional


'''
CURRENTLY USED INPUT PROMPT EXAMPLE (WITH REFERENCES TO IMAGES)

"You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.\n\n\
EXAMPLE SOLVED TASK:\n\n\
[User]\n\
This image (Input 1) changes to this one (Output 1).\n\
So how should this one (Input 2) change? Complete the pattern in this one (Output 2) to complete the puzzle.\n\n\
Input 1: [0 4 0] [0 4 0] [0 4 0]\n\
Output 1: [0 0 0] [0 4 0] [0 0 0]\n\
Input 2: [0 6 0] [0 6 0] [0 6 0]\n\
Output 2: \n\n\
[Assistant]\n\
[0 0 0] [0 6 0] [0 0 0]"
'''


'''
OPTIONAL INPUT PROMPT EXAMPLE (NO REFERENCES TO IMAGES)

"You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.\n\n\
EXAMPLE SOLVED TASK:\n\n\
[User]\n\
Input 1 changes to Output 1.\n\
So how should Input 2 change? Complete the pattern in Output 2 to complete the puzzle.\n\n\
Input 1: [0 4 0] [0 4 0] [0 4 0]\n\
Output 1: [0 0 0] [0 4 0] [0 0 0]\n\
Input 2: [0 6 0] [0 6 0] [0 6 0]\n\
Output 2: \n\n\
[Assistant]\n\
[0 0 0] [0 6 0] [0 0 0]"
'''


def gpt_call(
            input: str, 
            model: str, 
            system_prompt: str, 
            temperature: Optional[float] = 0.0
         ) -> str:
    '''
    Calls OPENAI API with input and system prompt.
    '''
    
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[

            {"role": "system",
            "content": system_prompt},

            {"role": "user", 
                "content": input},
                
        ],
    )

    return completion.choices[0].message.content

def together_call(
            input: str, 
            model: str, 
            temperature: Optional[float] = 0.0
        ) -> str:
    '''
    Calls TOGETHER API with input prompt.
    '''
    
    output = together.Complete.create(
        prompt = input, 
        model = model, 
        max_tokens = 256,
        temperature = temperature,
        stop = ['<human>', '\n\n']
    )

    return output['output']['choices'][0]['text']

def gpt_call_vision(
            input_text: str, 
            input_img: str,
            system_prompt_text: str,
            system_prompt_img: str = None
        ) -> str:
    '''
    Calls OPENAI API (GPT-4 Vision) with input as images and system prompt.
    '''
    if system_prompt_img is None:
        system_prompt_img = task_to_base64(0)  # Example task

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ["OPENAI_API_KEY"]}"
    }

    payload = {
            "model": "gpt-4-vision-preview",
            "messages": [

                    {"role": "system",
                    "content": [
                            
                            {"type": "text",
                            "text": system_prompt_text
                            },

                            {"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{system_prompt_img}"}
                            }
                    ]},

                    {"role": "user",
                    "content": [
                            
                            {"type": "text",
                            "text": input_text
                            },

                            {"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{input_img}"}
                            }
                    ]}
            ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']




if __name__ == '__main__':

    #*** INPUT PROMPTS ***

    system_prompt = "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.\n\n\
    EXAMPLE SOLVED TASK:\n\n\
    [User]\n\
    This image (Input 1) changes to this one (Output 1).\n\
    So how should this one (Input 2) change? Complete the pattern in this one (Output 2) to complete the puzzle.\n\n\
    Input 1: [0 4 0] [0 4 0] [0 4 0]\n\
    Output 1: [0 0 0] [0 4 0] [0 0 0]\n\
    Input 2: [0 6 0] [0 6 0] [0 6 0]\n\
    Output 2: \n\n\
    [Assistant]\n\
    [0 0 0] [0 6 0] [0 0 0]"

    system_prompt_vision_image = "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text. Provide your response using the following color pallette:\n\
    0: black\n\
    1: firebrick\n\
    2: darkorange\n\
    3: gold\n\
    4: teal\n\
    5: dodgerblue\n\
    6: rebeccapurple\n\
    7: hotpink\
    \n\nEXAMPLE SOLVED TASK:\n\n\
    [User]\n\
    This image (top left) changes to this one (top right).\n\
    So how should this one (bottom left) change? Complete the pattern in this one (bottom right) to complete the puzzle.\n\n\
    [Assistant]\n\
    [0 0 0] [0 6 0] [0 0 0]"

    system_prompt_vision_image_text = "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text. Provide your response using the following color pallette:\n\
    0: black\n\
    1: firebrick\n\
    2: darkorange\n\
    3: gold\n\
    4: teal\n\
    5: dodgerblue\n\
    6: rebeccapurple\n\
    7: hotpink\
    \n\nEXAMPLE SOLVED TASK:\n\n\
    [User]\n\
    This image (top left) changes to this one (top right).\n\
    So how should this one (bottom left) change? Complete the pattern in this one (bottom right) to complete the puzzle.\n\n\
    Input 1: [0 4 0] [0 4 0] [0 4 0]\n\
    Output 1: [0 0 0] [0 4 0] [0 0 0]\n\
    Input 2: [0 6 0] [0 6 0] [0 6 0]\n\
    Output 2: \n\n\
    [Assistant]\n\
    [0 0 0] [0 6 0] [0 0 0]"

    starting_str = "\n\nTEST TASK:\n\n" # Text before task input


    #*** GPT CALLS ***

    models = ['gpt-4-0613', 'gpt-3.5-turbo-0301']
    results_dict = {model: {str(task): {'response': [], 'correct': []} for task in range(1,17)} for model in models}


    # call models
    for model in models:
        print(f'_____ Model: {model} _____')

        for task in range(1,17):  
            response = gpt_call(
                input=task_to_input(task, input_str=starting_str), 
                model=model, 
                system_prompt=system_prompt
            )
                
            results_dict[model][str(task)]['response'].append(response)
            results_dict[model][str(task)]['correct'].append(check_response(task, response))
            print(f'Task {task} - Correct: {bool(results_dict[model][str(task)]['correct'][0])}')

        json.dump(results_dict, open('data/model/GPT.json', 'w'))

    # gpt vision
    prompts = ['system_prompt_vision_image', 'system_prompt_vision_image_text']
    results_dict = {prompt: {str(task): {'response': [], 'correct': []} for task in range(1,17)} for prompt in prompts}
    for prompt in prompts:
        for task in range(1,17):  
            response = gpt_call_vision(
                input_text = task_to_input(task, input_str=starting_str) if prompt == 'system_prompt_vision_image_text' else starting_str, 
                input_img = task_to_base64(task),
                system_prompt_text = eval(prompt)
            )
            
            results_dict[prompt][str(task)]['response'].append(response)
            results_dict[prompt][str(task)]['correct'].append(check_response(task, response))
            print(f'Task {task} - Correct: {bool(results_dict[prompt][str(task)]['correct'][0])}')

    json.dump(results_dict, open('data/model/GPT_vision.json', 'w'))

    
    #*** TOGETHER CALLS ***
            
    # get available models
    model_list = together.Models.list()

    # filter for chat models
    chat_model_list = [model for model in model_list if 'display_type' in model.keys() and model['display_type'] == 'chat']
    chat_model_names = [model['name'] for model in chat_model_list]

    # load results dict if it exists
    if 'TOGETHER.json' in os.listdir('data/model'):
        results_dict = json.load(open('data/model/TOGETHER.json', 'r'))
        # filter for models that have not been called yet
        chat_model_names = [model for model in results_dict if len(results_dict[model]['16']['response']) == 0]
    else:
        results_dict = {model: {str(task): {'response': [], 'correct': []} for task in range(1,17)} for model in chat_model_names}

    
    # call models
    for idx, model in enumerate(chat_model_names):
        print(f'_____ Model: {model} ({idx+1}/{len(chat_model_names)}) _____')

        for task in range(1,17):
            input = system_prompt + task_to_input(task, starting_str)

            try:
                response = together_call(
                    input=input,
                    model=model
                )
            except:
                print(f"Calling {model} failed")
                for task_NA in range(1, 17):
                    results_dict[model][str(task_NA)]['response'].append('NA')
                    results_dict[model][str(task_NA)]['correct'].append('NA')
                break
                
            results_dict[model][str(task)]['response'].append(response)
            results_dict[model][str(task)]['correct'].append(check_response(task, response))
            print(f'Task {task} - Correct: {bool(results_dict[model][str(task)]['correct'][0])}')

            # sleep to prevent rate limit
            time.sleep(1)
    
        json.dump(results_dict, open('data/model/TOGETHER.json', 'w'))
