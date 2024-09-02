#MAIN FILE
from transformers import (set_seed, GPT2LMHeadModel, GPT2Tokenizer, default_data_collator, GPT2Config, Pretrain, get_linear_schedule_with_warmup)
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn

# randomness is used in the model's initialization...
set_seed(42)

# I used both cpu and gpu in training
device = "cpu"

# Configuration of model, tokenizer and setting the pad token
model_name_or_path = "distilgpt2"
gpt2config = GPT2Config()
model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=gpt2config)

tokenizer_name_or_path = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# pad token is the eos token for all gpt2 models
tokenizer.pad_token_id = tokenizer.eos_token_id
pad = tokenizer.pad_token_id

"""gpt2config.hidden_size = 768
gpt2config.num_hidden_layers = 24
gpt2config.max_position_embeddings=1024
gpt2config.n_inner = None
gpt2config.n_embd = 768
gpt2config.num_attention_heads = 12
gpt2config.scale_attn_weights = True
gpt2config.scale_attn_by_inverse_layer_idx = False
gpt2config.reorder_and_upcast_attn = False
"""
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Write a complete resume point for the given job.",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "isashap/resume-dataset-w-context"
dataset = load_dataset("isashap/resume-dataset-w-context")
print(dataset["train"][0])
text_column = "text"
label_column = "label"
max_length = 64
lr = 3e-2
num_epochs = 10
batch_size = 8

# usage: 
  #processing all of the resume points into a string of tokens useable as input to GPT2
  #sentences are padded to the max length as to not get cut off in the middle
def preprocess_training_function(examples):
    #number of times for the text to run, 417 training and 62 for eval
    batch_size = len(examples[text_column])

   #text and label input, along with the context
    label_input = [f'Please provide a resume point using the following job and key words:<|endoftext|>{job}<|endoftext|>' for job in examples['job']]
    context_input = [f'Key words:<|endoftext|>{x.lower()}<|endoftext|>Resume Point:' for x in examples["context"]]
    text_input = [f'<|endoftext|>{x if x is not None else ""}<|endoftext|>' for x in examples[text_column]]
    
    #named label mask here, but it’s just the tokenized labels
    label_mask = tokenizer(label_input, truncation=True)
    model_inputs = tokenizer(text_input, truncation=True)
    context_ids = tokenizer(context_input, truncation=True)

    
    for i in range(batch_size):
        context_id = context_ids["input_ids"][i]
        text_input_id = model_inputs["input_ids"][i]
        label_input_id = label_mask["input_ids"][i]
    
        model_inputs["input_ids"][i] = label_input_id + context_id + text_input_id
    
        if model_inputs["input_ids"][i][-1] == pad:
            model_inputs["input_ids"][i].pop()
        if model_inputs["input_ids"][i][-1] != 13 and model_inputs["input_ids"][i][-1] != pad:
            model_inputs["input_ids"][i].append(13)
    
        attention_mask = []
        for token in model_inputs["input_ids"][i]:
            if token == pad:
                attention_mask.append(0)
            elif token != pad:
                attention_mask.append(1)
            else:
                print("error!! token not added to attention mask D:")
                raise AssertionError
    
        model_inputs["attention_mask"][i] = attention_mask
        #to add the end eos token in again
        #model_inputs["input_ids"][i].append(pad)
    
        label_mask["attention_mask"][i] = [0] * ((len(label_input_id) + 1) + len(context_id)) + [1] * (len(text_input_id) - 2)
    
        print("model attention mask:", model_inputs["attention_mask"][i])
        print("model label mask   ", label_mask["attention_mask"][i])
        print("model input ids    ", model_inputs["input_ids"][i])
        
        print("model attention mask:", model_inputs["attention_mask"][i])
        print("model label mask   ", label_mask["attention_mask"][i])
        print("model input ids    ", model_inputs["input_ids"][i])
        
        model_inputs["input_ids"][i] = model_inputs["input_ids"][i] + [tokenizer.pad_token_id] * (max_length - len(model_inputs["input_ids"][i]))
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [0] * (max_length - len(model_inputs["attention_mask"][i]))
        model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] + [-100] * (max_length - len(model_inputs["attention_mask"][i]))
        label_mask["attention_mask"][i] = label_mask["attention_mask"][i] + [-100] * (max_length - len(label_mask["attention_mask"][i]))
        
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        label_mask["attention_mask"][i] = torch.tensor(label_mask["attention_mask"][i][:max_length])
        
    model_inputs["label_mask"] = label_mask["attention_mask"]

    print("model attention mask tensor:", model_inputs["attention_mask"][i])
    print("model label mask tensor   ", model_inputs["label_mask"][i])
    print("model input ids tensors    ", model_inputs["input_ids"][i])
    
    return model_inputs

processed_datasets = dataset.map(
    preprocess_training_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

#training
def train_resume_point_generator(model, train_dataloader, optimizer, lr_scheduler, criterion, num_epochs=10, device="cuda"):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
        
            optimizer.zero_grad()
        
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label_mask = batch['label_mask']
        
            #for use with gpu
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_mask = label_mask.to(device)
        
            #target sequence should be the same as the input sequence,
            #just with a label mask covering the answers
            target_ids = input_ids.clone()
            target_ids[label_mask == 0] = -100
            
            #label_mask to target_ids, commented out since it wasnt 
            '''target_ids = input_ids.clone()
            target_ids[:, :-1] = input_ids[:, 1:]
            target_ids[:, -1] = -100  # Set the last token to -100 (ignore_index)
            
            #mod the attention_mask
            attention_mask[:, -1] = 0'''
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}")

        model.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for eval_batch in eval_dataloader:
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        
                eval_input_ids = eval_batch['input_ids']
                eval_attention_mask = eval_batch['attention_mask']
                eval_label_mask = eval_batch['label_mask']
                eval_input_ids = eval_input_ids.to(device)
                eval_attention_mask = eval_attention_mask.to(device)
                eval_label_mask = eval_label_mask.to(device)
                
                eval_target_ids = eval_input_ids.clone()
                eval_target_ids[eval_label_mask == 0] = -100
                
                eval_outputs = model(
                    input_ids=eval_input_ids,
                    attention_mask=eval_attention_mask,
                    labels=eval_target_ids
                )      
                eval_loss += eval_outputs.loss.item()

        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}: Validation Loss = {avg_eval_loss:.4f}")
        
    print("Training finished!")
      
crit = nn.CrossEntropyLoss(ignore_index=pad)
train_resume_point_generator(model, train_dataloader, optimizer, lr_scheduler, crit, num_epochs, device="cpu")

# model.save_pretrained('isashap/contexttrained')

tokenizer.save_pretrained('isashap/contexttrained')
model.push_to_hub("isashap/contexttrained-validationloss-waldomodel", use_auth_token=True)

