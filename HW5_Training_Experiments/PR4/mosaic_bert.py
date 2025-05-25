#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizer,
    DataCollatorForLanguageModeling, TextDataset
)
import numpy as np
from tqdm import tqdm
import wandb
import argparse
import pandas as pd
import json

class AgriculturalTextDataset(Dataset):
    """Датасет для текстовых данных о сельскохозяйственных рисках"""
    
    def __init__(self, data_path, risk_type="diseases", lang="ru", tokenizer=None, max_length=128):
        self.data_path = data_path
        self.risk_type = risk_type
        self.lang = lang
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Загружаем данные
        self.texts = self._load_data()
        
    def _load_data(self):
        texts = []
        
        # Определяем файлы в зависимости от типа риска
        if self.risk_type == "diseases":
            descriptions_file = os.path.join(self.data_path, "disease_descriptions.csv")
        elif self.risk_type == "pests":
            descriptions_file = os.path.join(self.data_path, "vermin_descriptions.csv")
        elif self.risk_type == "weeds":
            descriptions_file = os.path.join(self.data_path, "weed_descriptions.csv")
        else:
            raise ValueError(f"Unsupported risk type: {self.risk_type}")
        
        # Загружаем описания
        if os.path.exists(descriptions_file):
            descriptions = pd.read_csv(descriptions_file)
            
            # Выбираем текстовые поля в зависимости от языка
            lang_suffix = {"ru": "ru", "ua": "ua", "en": "en"}[self.lang]
            
            # Собираем все текстовые поля
            text_fields = [col for col in descriptions.columns if col.endswith(f"_{lang_suffix}")]
            
            # Объединяем все текстовые данные
            for _, row in descriptions.iterrows():
                for field in text_fields:
                    if pd.notna(row[field]) and len(str(row[field])) > 10:  # Проверка на не-пустые тексты
                        texts.append(str(row[field]))
            
        # Дополнительно загружаем JSON-файлы с данными
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json') and self.risk_type in filename:
                try:
                    with open(os.path.join(self.data_path, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Ищем текстовые данные в JSON
                        if isinstance(data, list):
                            for item in data:
                                self._extract_texts_from_json(item, texts)
                        elif isinstance(data, dict):
                            self._extract_texts_from_json(data, texts)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(texts)} texts for {self.risk_type} in {self.lang}")
        return texts
    
    def _extract_texts_from_json(self, item, texts_list):
        """Извлечение текстов из JSON структуры"""
        if isinstance(item, dict):
            # Находим текстовые поля
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 50:
                    texts_list.append(value)
                elif isinstance(value, (dict, list)):
                    self._extract_texts_from_json(value, texts_list)
        elif isinstance(item, list):
            for elem in item:
                self._extract_texts_from_json(elem, texts_list)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Токенизация если доступен токенизатор
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Убираем размерность батча
            return {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            return {"text": text}

def create_tokenizer(vocab_size=30522):
    """Создание токенизатора BERT"""
    return BertTokenizer.from_pretrained('bert-base-uncased')

def create_bert_config(vocab_size=30522, hidden_size=768, num_hidden_layers=6, num_attention_heads=12):
    """Создание конфигурации модели BERT"""
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=512,
    )

import os
import argparse
import torch
import yaml
import wandb
from datetime import datetime
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    BertTokenizerFast,
    AutoTokenizer,
    get_scheduler
)
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelArguments:
    """
    Аргументы, связанные с моделью
    """
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    hidden_size: int = field(
        default=768, metadata={"help": "Hidden size of the model"}
    )
    num_hidden_layers: int = field(
        default=12, metadata={"help": "Number of hidden layers"}
    )
    num_attention_heads: int = field(
        default=12, metadata={"help": "Number of attention heads"}
    )
    intermediate_size: int = field(
        default=3072, metadata={"help": "Intermediate size of the FFN"}
    )
    hidden_dropout_prob: float = field(
        default=0.1, metadata={"help": "Hidden dropout probability"}
    )
    attention_probs_dropout_prob: float = field(
        default=0.1, metadata={"help": "Attention dropout probability"}
    )
    max_position_embeddings: int = field(
        default=512, metadata={"help": "Maximum position embeddings"}
    )
    type_vocab_size: int = field(
        default=2, metadata={"help": "Type vocab size"}
    )
    initializer_range: float = field(
        default=0.02, metadata={"help": "Initializer range"}
    )
    vocab_size: int = field(
        default=30522, metadata={"help": "Vocabulary size"}
    )
    use_alibi: bool = field(
        default=True, metadata={"help": "Whether to use ALiBi positional encoding"}
    )
    use_flash_attention: bool = field(
        default=False, metadata={"help": "Whether to use Flash Attention"}
    )

@dataclass
class DataTrainingArguments:
    """
    Аргументы, связанные с данными и обучением
    """
    dataset_name: Optional[str] = field(
        default="wikitext", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="wikitext-103-raw-v1", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=500, metadata={"help": "Run evaluation every X steps."})
    output_dir: str = field(
        default="./mosaic_bert_output",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

class AlibiEmbeddings(torch.nn.Module):
    """
    Реализация ALiBi (Attention with Linear Biases) из статьи "Train Short, Test Long"
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.token_type_embeddings.weight.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def create_alibi_bert():
    """
    Создает модель BERT с ALiBi вместо позиционного кодирования
    """
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        position_embedding_type=None,  # Отключаем стандартное позиционное кодирование
    )

    model = BertForMaskedLM(config)

    # Заменяем стандартные эмбеддинги на ALiBi
    original_embeddings = model.bert.embeddings
    model.bert.embeddings = AlibiEmbeddings(config)

    # Копируем веса
    model.bert.embeddings.word_embeddings.weight.data = original_embeddings.word_embeddings.weight.data
    model.bert.embeddings.token_type_embeddings.weight.data = original_embeddings.token_type_embeddings.weight.data

    return model

def prepare_alibi_tensor(max_seq_length, num_heads, dtype=torch.float):
    """
    Подготовка тензора ALiBi для масштабирования внимания
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=dtype
    )

    powers = torch.arange(1, 1 + closest_power_of_2, dtype=dtype)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=dtype
        )
        extra_powers = torch.arange(1, 1 + (num_heads - closest_power_of_2), dtype=dtype)
        extra_slopes = torch.pow(extra_base, extra_powers)
        slopes = torch.cat([slopes, extra_slopes])

    # Создаем матрицу расстояний
    arange_tensor = torch.arange(max_seq_length, dtype=dtype)[None, :]
    alibi = -slopes[:, None, None] * (arange_tensor - arange_tensor.T).abs()

    return alibi.reshape(1, num_heads, max_seq_length, max_seq_length)

def patch_bert_for_alibi(model, args):
    """
    Патчит модель BERT для использования ALiBi
    """
    config = model.config

    # Подготовка тензора ALiBi
    alibi_tensor = prepare_alibi_tensor(
        args.max_seq_length,
        config.num_attention_heads,
        dtype=next(model.parameters()).dtype
    ).to(next(model.parameters()).device)

    # Функция для патчинга внимания
    def patch_self_attention_forward(module):
        original_forward = module.forward

        def forward_with_alibi(*args, **kwargs):
            # Получаем результаты оригинального forward
            outputs = original_forward(*args, **kwargs)

            # Добавляем тензор ALiBi к скорам внимания
            if 'attention_scores' in kwargs:
                batch_size = kwargs['attention_scores'].shape[0]
                alibi_for_batch = alibi_tensor.repeat(batch_size, 1, 1, 1)
                kwargs['attention_scores'] = kwargs['attention_scores'] + alibi_for_batch

            return outputs

        module.forward = forward_with_alibi

    # Патчим все модули внимания
    for layer in model.bert.encoder.layer:
        patch_self_attention_forward(layer.attention.self)

    return model

def create_optimized_bert(args):
    """
    Создает оптимизированную модель BERT согласно рекомендациям MosaicML
    """
    if args.use_alibi:
        model = create_alibi_bert()
        model = patch_bert_for_alibi(model, args)
    else:
        config = BertConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            initializer_range=args.initializer_range,
            vocab_size=args.vocab_size,
        )
        model = BertForMaskedLM(config)

    return model

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Инициализация Weights & Biases
    wandb.init(project="mosaic-bert-reproduction", config={**vars(model_args), **vars(data_args)})

    # Создаем каталог для результатов
    os.makedirs(data_args.output_dir, exist_ok=True)

    # Подготовка данных
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        raw_datasets = load_dataset("text", data_files=data_files)

    # Загрузка токенизатора
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Создание модели
    model = create_optimized_bert(model_args)

    # Токенизация и подготовка данных
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            return_special_tokens_mask=True,
        )
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=["text"],
        load_from_cache_file=True,
    )

    # Подготовка коллатора данных для маскированного языкового моделирования
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability,
    )

    # Настройка аргументов обучения
    training_args = TrainingArguments(
        output_dir=data_args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=data_args.num_train_epochs,
        per_device_train_batch_size=data_args.per_device_train_batch_size,
        per_device_eval_batch_size=data_args.per_device_eval_batch_size,
        learning_rate=data_args.learning_rate,
        weight_decay=data_args.weight_decay,
        warmup_steps=data_args.warmup_steps,
        logging_dir=f"{data_args.output_dir}/logs",
        logging_steps=data_args.logging_steps,
        save_steps=data_args.save_steps,
        evaluation_strategy="steps" if data_args.eval_steps > 0 else "no",
        eval_steps=data_args.eval_steps if data_args.eval_steps > 0 else None,
        report_to="wandb",
    )

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
    )

    # Обучение модели
    print("Starting training...")
    trainer.train()

    # Сохранение модели
    trainer.save_model()

    # Оценка модели
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Логирование результатов в W&B
    wandb.log({"perplexity": math.exp(eval_results['eval_loss'])})

    # Завершение эксперимента
    wandb.finish()

if __name__ == "__main__":
    main()
def create_bert_model(config):
    """Создание модели BERT для предобучения"""
    return BertForMaskedLM(config)

def prepare_data_loaders(dataset, batch_size=32, train_ratio=0.9):
    """Подготовка загрузчиков данных для обучения и валидации"""
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Создаем data collator для задачи MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    """Обучение модели на одну эпоху"""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # Перемещаем данные на устройство
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Обновляем learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Обновляем статистику
        running_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return running_loss / len(train_loader)

def evaluate(model, val_loader, device):
    """Оценка модели на валидационных данных"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Перемещаем данные на устройство
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Обновляем статистику
            running_loss += loss.item()
    
    return running_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description="Pretrain a small BERT model from scratch")
    parser.add_argument("--data_path", default="../crawler/downloads", help="Path to data directory")
    parser.add_argument("--risk_type", default="diseases", choices=["diseases", "pests", "weeds"], help="Type of risk")
    parser.add_argument("--lang", default="ru", choices=["ru", "ua", "en"], help="Language")
    parser.add_argument("--output_dir", default="models", help="Directory to save model")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # Инициализация W&B
    if not args.no_wandb:
        wandb.init(project="agri-risk-bert", config=vars(args))
    
    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Создаем токенизатор
    tokenizer = create_tokenizer()
    
    # Создаем датасет
    dataset = AgriculturalTextDataset(
        data_path=args.data_path,
        risk_type=args.risk_type,
        lang=args.lang,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Подготовка загрузчиков данных
    train_loader, val_loader = prepare_data_loaders(dataset, batch_size=args.batch_size)
    
    # Создаем конфигурацию BERT
    config = create_bert_config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads
    )
    
    # Создаем модель BERT
    model = create_bert_model(config)
    model = model.to(device)
    
    # Оптимизатор и scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Линейный scheduler с warmup
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    def get_lr_scheduler(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_scheduler)
    
    # Создаем директорию для сохранения моделей
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Обучение модели
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Обучение
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Валидация
        val_loss = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Логируем в W&B
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Сохраняем модель
            model_path = os.path.join(args.output_dir, f"mosaic_bert_{args.risk_type}_{args.lang}_best.pt")
            torch.save(model.state_dict(), model_path)
            
            # Сохраняем конфигурацию и токенизатор
            config_path = os.path.join(args.output_dir, f"mosaic_bert_{args.risk_type}_{args.lang}_config.json")
            with open(config_path, 'w') as f:
                f.write(config.to_json_string())
            
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"mosaic_bert_{args.risk_type}_{args.lang}_tokenizer"))
    
    # Сохраняем последнюю модель
    model_path = os.path.join(args.output_dir, f"mosaic_bert_{args.risk_type}_{args.lang}_last.pt")
    torch.save(model.state_dict(), model_path)
    
    # Логируем модель в W&B
    if not args.no_wandb:
        wandb.save(model_path)
        wandb.finish()
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
