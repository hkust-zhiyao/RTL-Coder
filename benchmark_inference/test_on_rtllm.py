from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
from transformers import AutoModelForSeq2SeqLM
import os
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
parser = argparse.ArgumentParser(description='Process some test.')
parser.add_argument('--model', type=str)
parser.add_argument('--temperature', type=float)
parser.add_argument('--gpu_name', type=int)
parser.add_argument('--n', type=int) # candidate num to each of the instruction
parser.add_argument('--output_dir', type=str)
args = parser.parse_args()
import json
import tqdm
import copy
design_list = ['accu', 'adder_8bit', 'adder_16bit', 'adder_32bit', 'adder_pipe_64bit', 'asyn_fifo', 'calendar', 'counter_12', 'edge_detect',
               'freq_div', 'fsm', 'JC_counter', 'multi_16bit', 'multi_booth_8bit', 'multi_pipe_4bit', 'multi_pipe_8bit', 'parallel2serial' , 'pe_single' , 'pulse_detect', 
               'radix2_div', 
               'RAM_single', 'right_shifter', 
               'serial2parallel', 
               'signal_generator','synchronizer', 'alu', 'div_16bit', 'traffic_light', 'width_8to16']
def load_testjson(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

bench_path = 'rtllm-1.1.json'
bench_data = load_testjson(bench_path)

progress_bar = tqdm.tqdm(total=len(bench_data) * args.n)
checkpoint = args.model
config = AutoConfig.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map=args.gpu_name)
model.eval()
gen_batch_size = 1 # inference batch size

output_dir = args.output_dir

for iter in range(args.n):
    id = 1
    if not os.path.exists(os.path.join(output_dir, "test_{}".format(iter+1))):
        os.mkdir(os.path.join(output_dir, "test_{}".format(iter+1)))
    save_path = os.path.join(output_dir, "test_{}".format(iter+1))

    while id <= len(bench_data):
        result_list = []
        tmp_list = []
        inp_list = []
        for ite in range(gen_batch_size):
            dic = bench_data[id - 1]
            prompt = dic['Instruction'] + '\n' + dic['Input'] + '\n'
            tmp_list.append(prompt)
            inp_list.append(dic['Input'] + '\n')
            id = id + 1
            if id > len(bench_data):
                break
        inputs = tokenizer(tmp_list, return_tensors="pt", padding='longest').to(args.gpu_name)
        outputs = model.generate(inputs=inputs.input_ids, max_length=len(inputs[0]) + 2048, do_sample=True, temperature=args.temperature, top_p=0.95, 
        attention_mask=inputs.attention_mask)
        for res_i, output in enumerate(outputs):
            s_full = tokenizer.decode(output[len(inputs[0]):].cpu().squeeze(), skip_special_tokens=True)
            #please note that the RTLCoder-deepseek-v1.1 version requires a different code extraction method
            #if len(s_full.split('endmodulemodule', 1)) == 2:
                #s = s_full.split('endmodulemodule', 1)[0] + "\n" + "endmodule"
            #else:
                #s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
            #if s.find('top_module') != -1:
                #s = s.split('top_module', 1)[0]
                #s = s.rsplit('endmodule', 1)[0] + "\n" + "endmodule"
            #index = s.rfind('tb_module')
            #if index == -1:
                #index = s.find('testbench')
            #if index != -1:
                #s_tmp = s[:index]
                #s = s_tmp.rsplit("endmodule", 1)[0] + "\n" + "endmodule"
            #If the RTLCoder version is based on Mistral, just use the following code extraction method.
            s = s_full.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

            index = s.rfind('tb_module')
            if index == -1:
                index = s.find('testbench')
            if index != -1:
                s_tmp = s[:index]
                s = s_tmp.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

            result_list.append(inp_list[res_i] + s)


        for result in result_list:
            for keyword in design_list:
                if keyword in result:
                    with open(os.path.join(save_path, '{}.v'.format(keyword)), 'w') as f:
                        f.write(result)
                    f.close()
                    break
        
        progress_bar.update(len(result_list))


        
