# %%
import argparse
import json,hashlib
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator, DataCollatorWithPadding
import transformers
from torch import nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftModel
from datasets import IterableDataset
import datasets
import numpy as np
import get_data
import torch
from torch.utils.data import DataLoader
import importlib
import pathlib
import re
from typing import List
try:
    import requests
except:
    pass

importlib.reload(get_data)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def to_gpu_tensor(batch, device=device):
    return {key: torch.tensor(value).to(device) for key, value in batch.items()}

parser = argparse.ArgumentParser(
    description="Example script",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
parser.add_argument('--question_filename', type=str, default="questions_example.json", help="question_filename")
parser.add_argument('--write_path', type=str, default="/content/drive/MyDrive/models/predictions", help="write_path")
parser.add_argument('--df_start_id', type=int, default=0, help="df_start_id")
parser.add_argument('--df_end_id', type=int, default=None, help="df_start_id")
parser.add_argument('--model_dir_name', type=str, default="815aace0d964e9dad6ee4899b9ce83efea759b914d0a352c4e0cce1ad052d481", help="model_dir_name")
parser.add_argument('--checkpoint_num', type=str, default="120", help="checkpoint_num")
parser.add_argument('--ground_truths_filename', type=str, default="ALL.json", help="checkpoint_num")
parser.add_argument('--scan_ans_windowsize', type=int, default=20, help="scan_ans_windowsize")
parser.add_argument('--requrl', type=str, default=None, help="invoke request")

args = parser.parse_args()
parserargs_dict = vars(args)

print(f"parserargs_dict is {parserargs_dict}")
writepath = pathlib.Path(parserargs_dict['write_path'])
model_base_dir = get_data.current_folder
for target_create_path in [writepath, model_base_dir]:
    target_create_path.mkdir(parents=True, exist_ok=True)

model_base_dir = model_base_dir/parserargs_dict['model_dir_name']
checkpoint_dir = model_base_dir/f"checkpoint-{parserargs_dict['checkpoint_num']}"
argp_dict = (model_base_dir/"argp_dict.json").read_text()
argp_dict = json.loads(argp_dict)
# %%
argp_dict['bnb_4bit_compute_dtype'] = torch.bfloat16 if argp_dict['bnb_4bit_compute_dtype']=='bf16' else torch.float32
if argp_dict['load_in_8bit'] in [0,None,False] and argp_dict['load_in_4bit'] in [0,None,False]:
    is_quantized = False
    quantization_config = None
else:
    is_quantized = True
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=argp_dict['load_in_8bit'],
        load_in_4bit=argp_dict['load_in_4bit'],
        llm_int8_has_fp16_weight=argp_dict['llm_int8_has_fp16_weight'],
        bnb_4bit_use_double_quant=argp_dict['bnb_4bit_use_double_quant'],
        bnb_4bit_compute_dtype=argp_dict['bnb_4bit_compute_dtype'],
        bnb_4bit_quant_type=argp_dict['bnb_4bit_quant_type'],
    )

if argp_dict['peft_r'] in [0,None,False] or argp_dict['peft_lora_alpha'] in [0,None,False]:
    is_peft = False
    peft_config = None
else:
    is_peft = True
    peft_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        inference_mode=False,
        r=argp_dict['peft_r'],
        lora_alpha=argp_dict['peft_lora_alpha'],
        lora_dropout=0.1
    )

model = AutoModelForQuestionAnswering.from_pretrained(
    checkpoint_dir,
    token=get_data.access_token,
    trust_remote_code=True,
    quantization_config=quantization_config,
    attn_implementation=argp_dict['attn_implementation'],
    device_map="auto",
)
if is_quantized:
    model = prepare_model_for_kbit_training(model)
if is_peft:
    model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=False)
print("model load complete")
# %%
drop_cols_1 = ['start_positions', 'end_positions']
drop_cols_2 = ['qid', 'question', 'category', 'context']

# %%
if True:

    prediction_batch_size = parserargs_dict['batch_size']
    get_data_instance = get_data.GetdataClass(model_name=argp_dict['model_name'])
    competition_dataframe = get_data_instance.get_competition_dataset_faq(join_truths=False, questions_filename=parserargs_dict['question_filename'])
    competition_dataframe = competition_dataframe.iloc[0:parserargs_dict['df_start_id']].sort_values(by=['qid'], ascending=False)
    testset = datasets.Dataset.from_pandas(competition_dataframe)

    model = model.to(device)
    iterds_validation = testset \
        .map(get_data_instance.markup_article_id, batched=False, with_indices=False)
    data_collator = DataCollatorWithPadding(tokenizer=get_data_instance.tokenizer)
    dataloader = DataLoader(
        iterds_validation.remove_columns(column_names=drop_cols_2),
        batch_size=prediction_batch_size,
        collate_fn=data_collator
    )
    iterds_validation = iterds_validation.batch(prediction_batch_size)

    match_doc_id_pattern = re.compile(r'(?:insurance|finance|faq)_\d+')
    def slice_list_by_list_of_int(srclist:List, indices:List)->List:
        returnlist = []
        for id in indices:
            returnlist.append(srclist[id])
        return returnlist
    def model_output_arrange(model_outputs, batchdata=None, scan_ans_windowsize_limit:int=25)->List:
        global match_doc_id_pattern
        func_batch_predictions_for_competition = []
        max_seq_length = model_outputs.start_logits.shape[1]
        print(f"model_outputs.start_logits shape is {model_outputs.start_logits.shape}")
        pred_probs = tuple(torch.nn.functional.softmax(model_output, dim=-1) for model_output in [model_outputs.start_logits, model_outputs.end_logits])
        pred_probs_argmax = tuple(prob.argmax(dim=-1) for prob in pred_probs)
        reverse_start_end_indices = pred_probs_argmax[0]>pred_probs_argmax[1]
        pred_pos = torch.stack(pred_probs_argmax, dim=-1)
        pred_pos[:,0][reverse_start_end_indices] = pred_probs_argmax[1][reverse_start_end_indices]
        pred_pos[:,1][reverse_start_end_indices] = pred_probs_argmax[0][reverse_start_end_indices]
        for row_doc_i in range(pred_pos.shape[0]):
            srctokens = batchdata['input_ids'][row_doc_i]
            start = pred_pos[row_doc_i,0]
            end = pred_pos[row_doc_i,1]
            incre_num_by = 0
            while True:
                span = torch.arange(start, end+1)
                span = slice_list_by_list_of_int(srctokens, span.tolist())
                span = get_data_instance.tokenizer.decode(span)
                matches = match_doc_id_pattern.findall(span)
                if matches or (start==0 and end>=max_seq_length) or incre_num_by>scan_ans_windowsize_limit:
                    break
                else:
                    incre_num_by += 5
                    start = 0 if (start - incre_num_by)<=0 else (start-incre_num_by)
                    end = 0 if (end + incre_num_by)<=0 else (end+incre_num_by)
            if not matches:
                pass
            else:
                pred_catg, pred_article_num = matches[0].split("_")
                func_batch_predictions_for_competition.append({
                    "qid": batchdata['qid'][row_doc_i],
                    "category": pred_catg,
                    "retrieve": pred_article_num,
                    "incre_num_by": incre_num_by
                })
                print(f"{batchdata['qid'][row_doc_i]} done!")
        # print(f"batch_predictions_for_competition {func_batch_predictions_for_competition}")
        return func_batch_predictions_for_competition

    print("dataloader complete")
    overallpredictions = []
    with torch.no_grad():
        for iter_i, (batch, orig) in enumerate(zip(dataloader,iterds_validation)):
            print(f"iter_i {iter_i}")
            inputs = {key: val.to(device) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            towrite_dicts = model_output_arrange(outputs, orig, parserargs_dict['scan_ans_windowsize'])
            overallpredictions.extend(towrite_dicts)
            target_file_name_suffix = f"qid{orig['qid'][0]}_iter{iter_i}_checkpoint{parserargs_dict['checkpoint_num']}.json"
            target_file_name = writepath/target_file_name_suffix
            with target_file_name.open('w', encoding='utf-8') as json_file:
                json.dump(towrite_dicts, json_file, ensure_ascii=False, indent=4)
            if parserargs_dict['requrl'] is not None:
                try:
                    reqdict = {'filename':target_file_name_suffix,'towrite_dicts':towrite_dicts}
                    requests.post(parserargs_dict['requrl'], json=reqdict)
                except Exception as e:
                    print(f"sending req failed for {e}")


    target_file_name = writepath/parserargs_dict['ground_truths_filename']
    overallpredictions = {"ground_truths":overallpredictions}
    with target_file_name.open('w', encoding='utf-8') as json_file:
        json.dump(overallpredictions, json_file, ensure_ascii=False, indent=4)