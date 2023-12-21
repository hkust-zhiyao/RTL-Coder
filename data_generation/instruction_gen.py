import time
import json
import os
import random
import re
import time
import string
from functools import partial
from multiprocessing import Pool
import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

evolv_dic = ['p_example.txt']  # list of mutation method
evo_type = evolv_dic[0] #choose a  mutation method

def encode_prompt_topic(prompt_instructions):
    prompt = open(evo_type).read() + "\n"
    prompt += prompt_instructions
    prompt += '\n#task#\n'
    return prompt

def encode_prompt(prompt_instructions):
    prompt = open(evo_type).read() + "\n"
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input) = task_dict["Instruction"], task_dict["Input"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def encode_prompt_uniq(prompt_instructions):
    prompt = open(evo_type).read() + "\n"
    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input) = task_dict["Instruction"], task_dict["Input"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        # prompt += '#given prompt#\n'
        prompt += '\n{Instruction}\n'
        prompt += instruction
        prompt += '\n{Input}\n'
        prompt += input
    prompt += '\n#Rewritten prompt#\n'
    return prompt

def post_process_gpt3_response_uniq(response):
    if response is None:
        return []
    raw_instructions = response["text"]
    if response["finish_reason"] == "length":
        return []
    try:
        tmp = raw_instructions.split('{Instruction}', 1)[1]
        ins = tmp.split('{Input}', 1)[0].strip()
        inp = tmp.split('{Input}', 1)[1].strip()
        if len(ins.split()) <= 3 or len(ins.split()) > 400:
            return []
        if not ins[0].isascii():
            return []
        ans = {"Instruction": ins, "Input": inp}
        return [ans]
    except:
        return  []
def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input):", inst)
        if len(splitted_data) != 5:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        instructions.append({"Instruction": inst, "Input": input})
    return instructions

def generate_instruction_following_data(
    output_dir="",
    output_file = 'vary_func_1.json',
    seed_tasks_path="seed_tasks.jsonl",
    num_instructions_to_generate=100,
    num_prompt_instructions=4,
    request_batch_size=1,

):
    seed_instruction_data = utils.load_json(seed_tasks_path)
    print(f"Loaded {len(seed_instruction_data)} seed instructions")
    request_idx = 0
    target_instruction_data = []
    if os.path.exists(os.path.join(output_dir, output_file)):
        target_instruction_data = utils.load_json(os.path.join(output_dir, output_file))
        print(f"Loaded {len(target_instruction_data)} target  instructions")

    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if target_instruction_data:
        progress_bar.update(len(target_instruction_data))
    all_instructions = [d["Instruction"] for d in seed_instruction_data] + [
        d["Instruction"] for d in target_instruction_data
    ]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]


    while len(target_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            prompt = open(evo_type).read() + "\n"
            batch_inputs.append(prompt)
        request_start = time.time()
        print("Calling openai...")
        results = utils.askGPT35(question=batch_inputs[0])
        request_duration = time.time() - request_start
        print(f'request took - {request_duration}')
        process_start = time.time()
        instruction_data = []
        for result in results:
            print(result)
            new_instructions = post_process_gpt3_response_uniq(result)
            instruction_data += new_instructions
        print(f'recieve {len(instruction_data)} output\n')
        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["Instruction"])
            rouge_scores = [rouge_scorer._score_lcs(new_instruction_tokens,item) for item in all_instruction_tokens]
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            target_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["Instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")

        utils.save_json(target_instruction_data, os.path.join(output_dir, output_file))

import openai
#please set your openai api


generate_instruction_following_data(output_dir="",
        seed_tasks_path="data_sample.json",
        output_file='new_instructions.json',
        num_instructions_to_generate=50,
        num_prompt_instructions=1,
        request_batch_size=1,)


