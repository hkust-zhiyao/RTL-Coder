from typing import cast
import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import torch
import transformers
import copy
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import Dataset
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
    BitsAndBytesConfig
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."},
    )
    compare_weight: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    continue_adapter: str = field(default=None)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_max_length=2048):
        super(ScoreDataset, self).__init__()
        logging.warning("Loading data...")
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.dic_temp = [json.loads(line.strip()) for line in lines]
        print('total num of data: {}'.format(len(self.dic_temp)))
        self.data_temp = [item['Instruction'] for item in self.dic_temp]
        self.data = []
        index = 0
        for data in self.data_temp:
            data_toked = _single_tokenize(data, tokenizer)
            if data_toked.shape[0] < model_max_length * 0.5:
                self.data.append(self.dic_temp[index])
            index += 1
        print('filtered num of data: {}'.format(len(self.data)))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

def _single_tokenize(text, tokenizer, max_len=None):
    if max_len is None:
        max_len = tokenizer.model_max_length
    toked = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
        )
    return toked['input_ids'][0]

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):

        idxs = []
        all_scores = []
        input_ids = []
        score_mask = []
        labels = []
        for idx, ins in enumerate(instances):

            ins = ins['input_ids'] # hack
            query = ins['Instruction']
            responses = ins['Response']
            scores = ins['Score']
            all_scores.append(scores)
            idxs.append([idx] * len(scores))

            query_input_ids = _single_tokenize(query, self.tokenizer, max_len=self.tokenizer.model_max_length)
            query_target = torch.LongTensor([IGNORE_INDEX] * (query_input_ids.shape[0]))
            dummy_target = torch.LongTensor([IGNORE_INDEX])
            for res in responses:
                r = res
                res_input_ids = _single_tokenize(r + self.tokenizer.eos_token, self.tokenizer, max_len=self.tokenizer.model_max_length-query_input_ids.shape[0]) # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(torch.cat((query_target, res_input_ids), dim=0))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(tokenizer=tokenizer, data_path=data_args.data_path, model_max_length=training_args.model_max_length)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class CompareTrainer(Trainer):

    def get_comp_loss(self, logits, labels, attention_mask, scores):
        #  this function gather the logits a one batch to calculate the comparing score using normalizing
        #logits shape: (batch, can, L, vocab)
        #labels shape: (batch, can, L)
        #attention_mask shape: (batch, can, L)
        #score: (batch, can)
        compare_loss_list = []
        for batch_id in range(logits.size(0)):
            prod = []
        #the first dimension of logits should be cand
            for i in range(logits.size(1)):
                mask = attention_mask[batch_id][i].unsqueeze(0)
                logit = logits[batch_id][i].unsqueeze(0)
                label = labels[batch_id][i].unsqueeze(0)
                loss = torch.nn.CrossEntropyLoss(reduction="none")(
                    logit[..., :-1, :].contiguous().view(-1, logit.size(-1)),
                    label[..., 1:].contiguous().view(-1),
                ).view(label.size(0), label.size(-1) - 1)
                loss = loss * mask[..., 1:].contiguous()
                loss = loss[:, -label.size(1):].sum(dim=1)
                prod.append(-loss/mask.sum(-1))
            prod_tensor = torch.stack(prod)
            prod_normalized = torch.exp(prod_tensor) / torch.sum(torch.exp(prod_tensor))
            # print('**************probability distribution**************')
            # print(prod_tensor)
            # print('**************normalized**************')
            # print(prod_normalized)
            comp_loss = self.compare_loss(scores=prod_normalized, rw_scores=scores[batch_id].unsqueeze(0))
            compare_loss_list.append(comp_loss)
        com_loss_collect = torch.stack(compare_loss_list)
        return torch.mean(com_loss_collect)

    def compare_loss(self, scores, rw_scores):
        cand = rw_scores.shape[1]
        new_scores = scores.reshape(-1, cand)
        diff = new_scores.unsqueeze(1) - new_scores.unsqueeze(-1)
        rw_diff = rw_scores.unsqueeze(1) - rw_scores.unsqueeze(-1)
        aval = torch.bitwise_and(rw_diff - 0.2 > 0, diff-0.3 < 0)
        return -(diff[aval]-0.3).sum()

    def compute_loss(self, model, inputs, return_outputs=False):
        

        
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs[0]
        batch_size, _ = torch.max(inputs['idxs'][:, 0], dim=0)
        batch_size = batch_size.cpu().detach().item() + 1
        can_num = int(logits.size(0) / batch_size)
        L = logits.size(1)
        vocab = logits.size(2) #vocab = 32000
        logits = logits.view(batch_size, can_num, L, vocab) # batch * cand * L * V
        logits_sft = logits[:,-1,:,:] #batch * L * V
        label_len = inputs.get("labels").size(-1)
        lable_reshape = inputs.get("labels").view(batch_size, can_num, label_len) # batch * cand * L
        lables_sft = lable_reshape[:,-1,:] #batch * L

        domain_list = []
        for i in range(batch_size):
            shift_logits = logits_sft[i,:,:].unsqueeze(0)
            shift_logits = shift_logits[..., :-1, :].contiguous()
            shift_labels = lables_sft[i,:].unsqueeze(0)
            shift_labels = shift_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, vocab)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss_temp = loss_fct(shift_logits, shift_labels)
            domain_list.append(loss_temp)
        domain_loss = torch.stack(domain_list)
        domain_loss = torch.mean(domain_loss)

        attention_mask = inputs['attention_mask'].view(batch_size, can_num, L)
        scores = inputs['scores'].view(batch_size, can_num)
        comp_loss = self.get_comp_loss(logits=logits, labels=lable_reshape, attention_mask=attention_mask, scores=scores)
        loss = (self.args.compare_weight * comp_loss + 1) * domain_loss

        return (loss, scores) if return_outputs else loss

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
    )
    model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        print('smart tokenizer')
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = CompareTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
