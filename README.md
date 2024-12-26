# MedChat

A fined-tuned LLM (Llama Model) on a medical conversation dataset using PEFT technique called QLoRA and SFT

Model Link : https://huggingface.co/yuktasarode/Llama-2-7b-chat-finetune/tree/main

Dataset Link : https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k

Metrics:

BLEU score: 89.86

METEOR score: 0.94

Perplexity: 2.69

ROUGE-1: 0.95

ROUGE-2: 0.92

ROUGE-L: 0.95

## 1. Freezing Base Model Layers
In the provided code, the base model is loaded using:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
```
This initializes the pre-trained model (e.g., Llama-2) in a frozen state. The parameters of the base model are not updated during training to:

1. Preserve the pre-trained knowledge.
2. Reduce computational overhead and memory requirements.

## 2. Adding LoRA Layers
The lightweight LoRA layers are added to specific parts of the model (e.g., attention layers). These layers are parameterized with:
1. Low-rank matrices (dimensionality controlled by lora_r).
2. A scaling factor (lora_alpha) that balances the LoRA contributions.
This is done using:
```python
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
```
The LoraConfig defines how LoRA modifies the model's layers. The trainable LoRA layers are injected into the frozen model to adapt it to the new dataset without updating the base parameters.

## 3. Loading Dataset and Tokenizer
The dataset and tokenizer are loaded and configured for training:
```python
dataset = load_dataset(dataset_name, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```
The dataset is tokenized using the pre-trained tokenizer, ensuring compatibility with the base model.

## 4. Setting Training Parameters
The training parameters are specified in TrainingArguments:

```python
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)
```
Notable configurations include:

1. Gradient accumulation for larger batch sizes.
2. Gradient checkpointing to save memory.
3. Specific optimizers and learning rates suitable for fine-tuning.

## 5. Using the Trainer
The SFTTrainer orchestrates the fine-tuning:
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
```
The peft_config is applied to inject LoRA layers into the frozen model.
Only the parameters of these LoRA layers are updated during training.
The dataset and tokenizer are used to create batches for training.

## Summary of Key Steps
1. The base model is frozen to retain its pre-trained knowledge.
2. LoRA layers are added to the model, which are small and efficient to train.
3. Training only updates LoRA layers while the base model remains unchanged.
4. Efficient optimization techniques (e.g., QLoRA, gradient accumulation) are employed to reduce computational costs.
