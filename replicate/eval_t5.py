import glob
import json
import logging
import random
import re
import os
import csv
import argparse
from itertools import chain
from string import punctuation

import textwrap
from tqdm.auto import tqdm
from sklearn import metrics

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import csv
from dataclasses import dataclass
from enum import Enum
from typing import List
from typing import Optional
from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class T5FineTuner(pl.LightningModule):
  def __init__(self, hparams):
    super(T5FineTuner, self).__init__()
    self.hparams = hparams
    
    self.model = T5ForConditionalGeneration.from_pretrained(hparams['model_name_or_path'])
    self.tokenizer = T5Tokenizer.from_pretrained(hparams['tokenizer_name_or_path'])
  
  def is_logger(self):
    return self.trainer.proc_rank <= 0
  
  def forward(
      self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
  ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

  def _step(self, batch):
    lm_labels = batch["target_ids"]
    lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

    outputs = self(
        input_ids=batch["source_ids"],
        attention_mask=batch["source_mask"],
        lm_labels=lm_labels,
        decoder_attention_mask=batch['target_mask']
    )

    loss = outputs[0]

    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}
  
  def training_epoch_end(self, outputs):
    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}
  
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

  def configure_optimizers(self):
    "Prepare optimizer and schedule (linear warmup and decay)"

    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.hparams.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    self.opt = optimizer
    return [optimizer]
  
  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp: bool = False):
    if self.trainer.use_tpu:
      xm.optimizer_step(optimizer)
    else:
      optimizer.step()
    optimizer.zero_grad()
    self.lr_scheduler.step()
  
  def get_tqdm_dict(self):
    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

    return tqdm_dict

  def train_dataloader(self):
    train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
    dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
    t_total = (
        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        // self.hparams.gradient_accumulation_steps
        * float(self.hparams.num_train_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return dataloader

  def val_dataloader(self):
    val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="val", args=self.hparams)
    return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)



logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))



@dataclass(frozen=True)
class InputExample:
    idx: str
    question: str
    passage: str
    label: Optional[str]

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class DataProcessor:
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_source_examples(self, data_dir):
        raise NotImplementedError()
    def get_follow_examples(self, data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()

class BoolqProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")
    def get_source_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "source.csv")), "dev")
    def get_follow_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "follow-up.csv")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For BoolQ testing, the input file does not contain a label column. It can not be tested in current code setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        return ["False", "True"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                idx=line[0],
                question=line[1],
                passage=line[2],
                label=line[3],
            )
            for line in lines[1:]
        ]

        return examples

class BoolqDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=256):
    self.data_dir = data_dir
    self.type_path = type_path
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []
    self.proc = BoolqProcessor()
    self._build()
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze() 
    target_mask = self.targets[index]["attention_mask"].squeeze() 

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def __len__(self):
    return len(self.inputs)
  
  def _build(self):
    if self.type_path == 'train':
      examples = self.proc.get_train_examples(self.data_dir)
    elif self.type_path == 'source':
      examples = self.proc.get_source_examples(self.data_dir)
    elif self.type_path == 'follow-up':
      examples = self.proc.get_follow_examples(self.data_dir)
    
    for example in examples:
      self._create_features(example)
  
  def _create_features(self, example):
    question = example.question
    passage = example.passage
    input_ = "boolq question: %s  passage: %s </s>" % (question, passage)
    target = "%s </s>" % str(example.label)

    tokenized_inputs = self.tokenizer.batch_encode_plus(
        [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
    )
    tokenized_targets = self.tokenizer.batch_encode_plus(
        [target], max_length=6, pad_to_max_length=True, return_tensors="pt"
    )

    self.inputs.append(tokenized_inputs)
    self.targets.append(tokenized_targets)


def get_dataset(tokenizer, type_path, args):
  return BoolqDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mr",
        default='1-1',
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_dir",
        default='./data',
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        default='./data',
        type=str,
        required=True
    )

    args1 = parser.parse_args()

    file_path = args1.data_dir+'/source.jsonl'
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    question = []
    passage = []
    idx = []
    label = []
    for line in lines:
        i = json.loads(line.strip("\n"))
        question.append(i["question"])
        passage.append(i["passage"])
        idx.append(i["idx"])
        label.append(i["label"])
    f = open(args1.data_dir+'/source.csv','w', encoding="utf-8")
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(("idx", "question", "passage", "label"))
    for i in range(len(idx)):
        writer.writerow((idx[i], question[i], passage[i], label[i]))
    f.close()

    file_path = args1.data_dir+'/follow-up.jsonl'
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    question = []
    passage = []
    idx = []
    label = []
    for line in lines:
        i = json.loads(line.strip("\n"))
        question.append(i["question"])
        passage.append(i["passage"])
        idx.append(i["idx"])
        label.append(i["label"])
    f = open(args1.data_dir+'/follow-up.csv','w', encoding="utf-8")
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(("idx", "question", "passage", "label"))
    for i in range(len(idx)):
        writer.writerow((idx[i], question[i], passage[i], label[i]))
    f.close()

    
    set_seed(42)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5FineTuner.load_from_checkpoint(checkpoint_path=args1.output_dir+"/checkpointepoch=0.ckpt")

    output_sources = []
    output_follows = []

    dataset =  BoolqDataset(tokenizer, data_dir=args1.data_dir, type_path='source')
    loader = DataLoader(dataset, batch_size=8, num_workers=4)
    model.model.eval()
    outputs = []
    targets = []
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'], 
                                  attention_mask=batch['source_mask'], 
                                  max_length=6)

        dec = [tokenizer.decode(ids) for ids in outs]
        target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)
    print(metrics.accuracy_score(targets, outputs))
    output_sources = outputs

    dataset =  BoolqDataset(tokenizer, data_dir=args1.data_dir, type_path='follow-up')
    loader = DataLoader(dataset, batch_size=8, num_workers=4)
    model.model.eval()
    outputs = []
    targets = []
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'], 
                                  attention_mask=batch['source_mask'], 
                                  max_length=6)

        dec = [tokenizer.decode(ids) for ids in outs]
        target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)
    print(metrics.accuracy_score(targets, outputs))
    output_follows = outputs

    violations = []
    num = 0
    for source in output_sources:
        follow = output_follows[num]
        if source != follow:
            violations.append(source)
        num += 1
    
    if args1.mr in ["1-1", "1-2", "1-3", "1-4"]:
        print("violation rate: ", 1-len(violations)/len(output_sources))
    elif args1.mr in ["2-1", "2-2", "2-3"]:
        print("violation rate: ", len(violations)/len(output_sources))
    
    
    os.remove(args1.data_dir+'/source.csv')
    os.remove(args1.data_dir+'/follow-up.csv')


if __name__ == "__main__":
    main()