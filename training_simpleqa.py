# %%
import argparse
import ast
import json,hashlib
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer
from transformers import DefaultDataCollator, DataCollatorWithPadding
import transformers
from torch import nn
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import IterableDataset
import datasets
from sklearn.model_selection import StratifiedKFold
import numpy as np
import evaluate
import get_data
import torch
from accelerate import Accelerator
import multiprocessing
import importlib
import pathlib
import copy
from huggingface_hub import login as hflogin

importlib.reload(get_data)
# https://github.com/bdashore3/flash-attention/releases
# https://github.com/Dao-AILab/flash-attention/releases

googlecolab_output_dir = pathlib.Path("/content/drive/MyDrive/models")
kaggle_output_dir = pathlib.Path("/kaggle/working")
outputdir = googlecolab_output_dir if googlecolab_output_dir.exists() else (kaggle_output_dir/"models" if kaggle_output_dir.exists() else get_data.current_folder/"outputdir")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_collator = DefaultDataCollator()
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
skf = StratifiedKFold(n_splits=5)

def to_gpu_tensor(batch, device=device):
    return {key: torch.tensor(value).to(device) for key, value in batch.items()}

drop_cols_1 = ['qid','category','generate']
drop_cols_2 = ['answer_in_articleid','question','context','overflow_to_sample_mapping']
# %%


# def test_compute_metrics(*args, **kwargs):
#     # pred is the model outputs contains start_logits and end_logits
#     # truth is of size(n, 1)
#     global globaled_EvalPrediction
#     global globaled_EvalPrediction_dict
#     # answer_start_index = outputs.start_logits.argmax()
#     # answer_end_index = outputs.end_logits.argmax()
#     globaled_EvalPrediction = args
#     globaled_EvalPrediction_dict = kwargs
#     return {'f1':1,'ExactMatch':1}

def yield_output_span(t_start, t_end):
    output_tensors = [torch.arange(start, end+1) for start, end in zip(t_start, t_end)]
    return output_tensors
# %%
if True: #__name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Example script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")
    parser.add_argument('--max_steps_multiplier', type=float, default=1.0, help="max_steps_multiplier")
    parser.add_argument('--tpu', type=bool, default=False, help="using TPU")

    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning_rate")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="gradient_accumulation_steps")
    parser.add_argument('--torch_empty_cache_steps', type=int, default=2, help="torch_empty_cache_steps")
    # parser.add_argument('--peft_config', type=bool, default=True, help="peft_config")
    parser.add_argument('--peft_r', type=int, default=4, help="peft_r")
    parser.add_argument('--peft_lora_alpha', type=int, default=32, help="peft_lora_alpha")
    parser.add_argument('--torch_compile', type=ast.literal_eval, default=False, help="torch_compile")
    parser.add_argument('--fp16', type=ast.literal_eval, default=True, help="fp16")
    parser.add_argument('--fp16_opt_level', type=str, default="O1", help="fp16_opt_level")
    # parser.add_argument('--quantization_config', type=bool, default=False, help="quantization_config")

    parser.add_argument('--load_in_4bit', type=ast.literal_eval, default=False, help="load_in_4bit")
    parser.add_argument('--load_in_8bit', type=ast.literal_eval, default=False, help="load_in_8bit")
    parser.add_argument('--llm_int8_has_fp16_weight', type=ast.literal_eval, default=False, help="llm_int8_has_fp16_weight")
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="fp4", help="bnb_4bit_quant_type")
    parser.add_argument('--bnb_4bit_use_double_quant', type=ast.literal_eval, default=False, help="bnb_4bit_use_double_quant")
    parser.add_argument('--bnb_4bit_compute_dtype', type=str, default="bf16", help="bnb_4bit_compute_dtype")
    
    parser.add_argument('--optim', type=str, default="adamw_bnb_8bit", help="load_in_8bit")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight_decay")
    parser.add_argument('--attn_implementation', type=str, default="sdpa", help="attn_implementation") #flash_attention_2
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="model_name")
    parser.add_argument('--train_resume_from_checkpoint', type=str, default=None, help="train_resume_from_checkpoint")
    

    args = parser.parse_args()
    argp_dict = vars(args)
    # print(f"argp_dict: {argp_dict}")
    argp_dict_json_str = json.dumps(argp_dict, sort_keys=True)
    hash_object = hashlib.sha256(argp_dict_json_str.encode())
    hash_string = hash_object.hexdigest()
    final_output_dir = outputdir/"final"
    outputdir = outputdir/hash_string
    print(f"outputdir is {outputdir}")
    print(f"outputdir resolved is {outputdir.resolve()}")

    def compute_metrics(src_eval_pred:transformers.EvalPrediction):
        global outputdir
        outputdir = outputdir
        is_return_tensor = isinstance(
            src_eval_pred.predictions[0],
            torch.Tensor
        )
        compute_src = {}
        for attr in ['label_ids','predictions']:
            target_item = getattr(src_eval_pred, attr)
            if not is_return_tensor:
                compute_src[attr] = tuple(torch.tensor(target_item[i]) for i in range(len(target_item)) )
            else:
                compute_src[attr] = target_item
        true_pos = torch.stack(compute_src['label_ids'], axis=-1)
        pred_probs = tuple(torch.nn.functional.softmax(srct, dim=-1) for srct in compute_src['predictions'])
        pred_probs_argmax = tuple(prob.argmax(dim=-1) for prob in pred_probs)
        reverse_start_end_indices = pred_probs_argmax[0]>pred_probs_argmax[1]
        pred_pos = torch.stack(pred_probs_argmax, dim=-1)
        pred_pos[:,0][reverse_start_end_indices] = pred_probs_argmax[1][reverse_start_end_indices]
        pred_pos[:,1][reverse_start_end_indices] = pred_probs_argmax[0][reverse_start_end_indices]
        exact_matches = torch.eq(true_pos, pred_pos).all(dim=-1)
        pred_pos_spans = yield_output_span(pred_pos[:,0], pred_pos[:,1])
        true_pos_spans = yield_output_span(true_pos[:,0], true_pos[:,1])
        # f1_scores = []
        precision_scores = []
        recall_scores = []
        for row_doc_i, predspan in enumerate(pred_pos_spans):
            predspan_set = set(predspan.tolist())
            truthspan_set = set(true_pos_spans[row_doc_i].tolist())
            intersection = predspan_set&truthspan_set
            if len(predspan_set)!=0:
                precision = len(intersection)/len(predspan_set)
                precision_scores.append(precision)
            if len(truthspan_set)!=0:
                recall = len(intersection)/len(truthspan_set)
                recall_scores.append(recall)        
            # f1 = 2 * (precision * recall) / (precision + recall)
            # f1_scores.append(f1)
        metrics_exact_matches = exact_matches.mean(dtype=torch.float16).item()
        metrics_losses = src_eval_pred.losses.mean().item()
        returndict = {
            'precision':sum(precision_scores)/len(precision_scores),
            'recall':sum(recall_scores)/len(recall_scores),
            'ExactMatch':metrics_exact_matches,
            'loss':metrics_losses,
        }
        # metric_pathname_suffix = 0
        # while True:
        target_metric_path = outputdir / f"compute_metrics_0.json"
        if not target_metric_path.exists():
            with open(outputdir/target_metric_path, 'w') as f:
                json.dump(returndict, f)
        return returndict

    for odir in [final_output_dir,outputdir]:
        if not odir.exists():
            odir.mkdir(parents=True, exist_ok=True)
        # print(f"Folder '{outputdir}' created.")
    outputdir_str = str(outputdir.resolve())
    with open(outputdir/"argp_dict.json", 'w') as f:
        write_argdict_res = json.dump(argp_dict, f)
        # print(f"write_argdict_res is {write_argdict_res}")
    # print(f"outputdir_str is {outputdir_str}")
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
        argp_dict['model_name'],
        token=get_data.access_token,
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation=argp_dict['attn_implementation'],
        device_map="auto",
    )
    if is_quantized:
        model = prepare_model_for_kbit_training(model)
    if is_peft:
        model = get_peft_model(model, peft_config)
    if argp_dict['tpu']:
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        model.train()
        WRAPPED_MODEL = xmp.MpModelWrapper(model)
    # r=8, lora_alpha=32, trainable params: 542466 || all params: 494577028 || trainable%: 0.11
    # r=16, trainable params: 1083138 || all params: 495117700 || trainable%: 0.22
    # lora_alpha=64, trainable params: 542466 || all params: 494577028 || trainable%: 0.11
    print_trainable_parameters(model)
    get_data_instance = get_data.GetdataClass(model_name=argp_dict['model_name'])
    competition_dataframe = get_data_instance.get_competition_dataset_faq()
    try:
        hflogin(token=get_data.access_token)
    except Exception as e:
        print("hflogin failed due to {e}")

    for i, (train_index, test_index) in enumerate(skf.split(competition_dataframe, competition_dataframe['category'])):
        print(f"Fold {i}:")
        gen_train = get_data.EndlessGenerator(competition_dataframe.iloc[train_index])
        validationset = datasets.Dataset.from_pandas(competition_dataframe.iloc[test_index])
        print(f"  Train: {len(gen_train)} examples")
        print(f"  Test: {len(validationset)} examples")
        iterds_train = IterableDataset.from_generator(gen_train) \
            .remove_columns(column_names=drop_cols_1) \
            .map(get_data_instance.markup_article_id, batched=False, with_indices=False) \
            .remove_columns(column_names=drop_cols_2)
        iterds_validation = validationset \
            .remove_columns(column_names=drop_cols_1) \
            .map(get_data_instance.markup_article_id, batched=False, with_indices=False) \
            .remove_columns(column_names=drop_cols_2)

        data_collator = DataCollatorWithPadding(tokenizer=get_data_instance.tokenizer)

        one_epoch_steps = int(len(gen_train)/argp_dict['batch_size'])
        total_steps = int(one_epoch_steps*argp_dict['max_steps_multiplier'])
        training_args = TrainingArguments(
            output_dir=outputdir_str,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            learning_rate=argp_dict['learning_rate'],
            per_device_train_batch_size=argp_dict['batch_size'],
            per_device_eval_batch_size=argp_dict['batch_size'],
            gradient_accumulation_steps=argp_dict['gradient_accumulation_steps'],
            gradient_checkpointing=True,
            torch_empty_cache_steps=argp_dict['torch_empty_cache_steps'],
            fp16=True,
            fp16_opt_level=argp_dict['fp16_opt_level'],
            bf16=False,
            optim=argp_dict['optim'],
            dataloader_pin_memory=True,
            dataloader_num_workers=get_data.num_workers, #multiprocessing.cpu_count()//5,
            max_steps=total_steps,
            save_strategy='steps',
            save_steps=one_epoch_steps,
            save_total_limit=5,
            push_to_hub=True,
            hub_strategy="all_checkpoints",
            hub_token=get_data.access_token,
            hub_model_id="dowba/financial_rag_llm_qa",
            metric_for_best_model="eval_loss",
            include_for_metrics=['loss'],
            greater_is_better=True,
            load_best_model_at_end=True,
            logging_strategy='steps',
            logging_steps=one_epoch_steps,
            eval_strategy='steps',
            eval_steps=one_epoch_steps,
            fp16_full_eval=True,
            weight_decay=argp_dict['weight_decay'],
            restore_callback_states_from_checkpoint=True,
            resume_from_checkpoint=argp_dict['train_resume_from_checkpoint'],
            torch_compile=argp_dict['torch_compile'],
            report_to=['tensorboard'],
            auto_find_batch_size=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=iterds_train,
            eval_dataset=iterds_validation,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            processing_class=get_data_instance.tokenizer,
        )

        if True:
            resume_from_checkpoint = True if argp_dict['train_resume_from_checkpoint'] == 'True' else (None if argp_dict['train_resume_from_checkpoint'] in ['None',None] else argp_dict['train_resume_from_checkpoint'])
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        break
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str( final_output_dir.resolve()  ))
        merged_model_saved = True
        loramodel_saved = False
        print(f"merged_model_saved {merged_model_saved}")
    except Exception as e:
        merged_model_saved = False
        loramodel_saved = False
        print(f"merged_model_saved {merged_model_saved} due to {e}")
    if merged_model_saved == False:
        try:
            model.save_pretrained(str( final_output_dir.resolve() ))
            loramodel_saved = True
            print(f"loramodel_saved {loramodel_saved}")
        except Exception as e:
            loramodel_saved = False
            print(f"loramodel_saved {loramodel_saved} due to {e}")

    try:
        to_push_model = merged_model if merged_model_saved else model
        to_push_model.push_to_hub("financial_rag_llm_qa", token=get_data.access_token)
    except Exception as e:
        print(f"model push to hub error due to {e}")
    
    # %%
    # from evaluate import evaluator
    # from transformers import QuestionAnsweringPipeline
    # from get_data import tokenizer
    # globaled_EvalPrediction = None
    # globaled_EvalPrediction_dict = None
    # https://huggingface.co/docs/transformers/main_classes/trainer
    # https://huggingface.co/learn/nlp-course/chapter7/7

    # oracle = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)
    # print(
    #     oracle.create_sample(question="how old am i",context="i am 15 yrs old.")
    # )


    # task_evaluator = evaluator("question-answering")
    # results = task_evaluator.compute(
    #     metric="squad",
    # )