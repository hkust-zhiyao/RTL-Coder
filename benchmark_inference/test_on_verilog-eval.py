import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
import argparse
import json
import os
from tqdm import*

def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

parser = argparse.ArgumentParser(description='Process some test.')
parser.add_argument('--model', type=str)
parser.add_argument('--temperature', type=float)
parser.add_argument('--output_file', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--bench_type', type=str) # it can be Machine or Human
parser.add_argument('--gpu_name', type=int)
parser.add_argument('--n', type=int) # 'n' represent how many code candidates generated for each instruction
args = parser.parse_args()


descri_path = 'verilog-eval/descriptions/VerilogDescription_' + args.bench_type + '.jsonl'
input_path = 'verilog-eval/data/VerilogEval_' + args.bench_type + '.jsonl'

des_data = load_json(descri_path)
input_data = load_json(input_path)
progress_bar = tqdm.tqdm(total=len(des_data) * args.n)
tmp_list = des_data
des_data = []
for i in range(args.n):
    des_data += tmp_list


tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=args.gpu_name,)
model.eval()

id = 1

while id <= len(des_data):
    gen_batch_size = 1
    tmp_list = []
    dic_list = []
    for ite in range(gen_batch_size):
        item = des_data[id - 1]
        dic = {}
        dic['task_id'] = item['task_id']
        dic['description'] = item['detail_description']

        for j in range(len(input_data)):
            if input_data[j]['task_id'] == dic['task_id']:                
                dic['prompt'] = input_data[j]['prompt']
                break

        prompt = dic['description'] + '\n' + dic['prompt'] + '\n'
        inputs = tokenizer(prompt, return_tensors="pt", padding='longest').to(args.gpu_name)
        outputs = model.generate(inputs=inputs.input_ids, max_length=len(inputs[0]) + 1024, do_sample=True, temperature=args.temperature, top_p=0.95, attention_mask=inputs.attention_mask)
        s_full=tokenizer.decode(outputs[0][len(inputs[0]):].cpu().squeeze(), skip_special_tokens=True)

        s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
        index = s.rfind('tb_module')
        if index == -1:
            index = s.find('testbench')
        if index != -1:
            s_tmp = s[:index]
            s = s_tmp.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

        dic["output"]=s # outputs written to json results
        tmp_list.append(prompt)
        dic_list.append(dic)
        with open(os.path.join(args.output_dir, args.output_file),'a') as f:
            ob = json.dumps(dic)
            f.write(ob)
            f.write('\n')
        progress_bar.update(len(dic_list))
        id = id + 1
        if id > len(des_data):
            flag = 1
            break


    # for res_i, output in enumerate(outputs):
    #     s_full = tokenizer.decode(output[len(inputs[0]):].cpu().squeeze(), skip_special_tokens=True)
    #     #please note that the RTLCoder-deepseek-v1.1 version requires a different extraction method
    #     #if len(s_full.split('endmodulemodule', 1)) == 2:
    #         #s = s_full.split('endmodulemodule', 1)[0] + "\n" + "endmodule"
    #     #else:
    #         #s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
    #     #if s.find('top_module') != -1:
    #         #s = s.split('top_module', 1)[0]
    #         #s = s.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
    #     #index = s.rfind('tb_module')
    #     #if index == -1:
    #         #index = s.find('testbench')
    #     #if index != -1:
    #         #s_tmp = s[:index]
    #         #s = s_tmp.rsplit("endmodule", 1)[0] + "\n" + "endmodule"
    #     #If the RTLCoder version is based on Mistral, just use the following extraction method.
    #     ####
    #     s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

    #     # the model may output testbench after the design code
    #     index = s.rfind('tb_module')
    #     if index == -1:
    #         index = s.find('testbench')
    #     if index != -1:
    #         s_tmp = s[:index]
    #         s = s_tmp.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
    #     #####
    
