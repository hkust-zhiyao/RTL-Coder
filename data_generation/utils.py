import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union
import openai
import tqdm
from openai import openai_object
import copy




def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def save_json(dic_list, path):
    with open(path, 'w') as f:
        for dic in dic_list:
            ob = json.dumps(dic)
            f.write(ob)
            f.write('\n')


def askGPT35(question ,model='gpt-35-turbo', is_response=False, temperature=0.7):
    sleep_time = 2
    if is_response is True:
        p_message = [
            {'role': 'system', 'content': 'I want you act as a Professional Verilog coder.'},
            {'role': 'user', 'content': question}
        ]
    else:
        p_message = [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': question}
        ]
    max_gen_tokens = 2048
    count = 0
    while True:
        if count == 5:
            dic = {}
            dic['finish_reason'] = 'length'
            dic['text'] = ''
            return [dic]
        try:
            response = openai.ChatCompletion.create(
                engine=model,
                messages=p_message,
                temperature=temperature,
                max_tokens=max_gen_tokens,
            )
            ans = response['choices'][0]['message']['content']
            dic = {'text': ans, 'finish_reason': response['choices'][0]['finish_reason']}
            break

        except openai.error.OpenAIError as e:
            if 'maximum context' in str(e):
                count += 1 
                max_gen_tokens = int(max_gen_tokens / 1.3)
            logging.warning(f"OpenAIError: {e}.")
            logging.warning("Hit request rate limit; retrying...")
            time.sleep(sleep_time)
    return [dic]
