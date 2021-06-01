import json
import argparse
import csv
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME, AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.modeling_auto import AutoModelForMultipleChoice

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}

logger = logging.getLogger(__name__)


class BoolqExample(object):

    def __init__(self, boolq_id, context_sentence, start_ending, ending_0, label=None):
        self.boolq_id = boolq_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        attributes = [
            "boolq_id: {}".format(self.boolq_id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
        ]

        if self.label is not None:
            attributes.append("label: {}".format(self.label))

        return ", ".join(attributes)

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


def read_boolq_examples(input_file, is_training=True):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    examples = []
    for line in lines:
        data_raw = json.loads(line.strip("\n"))
        if data_raw["label"]:
            label_input = 1
        else:
            label_input = 0
        examples.append(
            BoolqExample(
                boolq_id=data_raw["idx"],
                context_sentence=data_raw["passage"],
                start_ending=data_raw["question"],
                ending_0="",
                label=label_input if is_training else None,
            )
        )

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    features = []
    for example_index, example in tqdm(enumerate(examples)):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        context_tokens_choice = context_tokens[:]
        ending_tokens = start_ending_tokens
        _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

        tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
        segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label = example.label
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("boolq_id: {}".format(example.boolq_id))
            logger.info("tokens: {}".format(" ".join(tokens)))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, input_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, segment_ids))))
            if is_training:
                logger.info("label: {}".format(label))

        features.append(InputFeatures(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids, label = label ))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels , output):
    outputs = (out.squeeze(1) >= 0.5).astype(np.int)
    for case in outputs:
        output.append(case)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, train = False, evaluate=False, test = False, output_examples=False, s_or_f = False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if evaluate:
        if s_or_f:
            input_file = args.data_file+"/source.jsonl"
        else:
            input_file = args.data_file+"/follow-up.jsonl"

    cached_features_file = os.path.join(
        os.path.dirname(input_file),
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_boolq_examples(input_file)
        features = convert_examples_to_features(examples, tokenizer, args.max_seq_length, not evaluate)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)

    if evaluate:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

    if output_examples:
        return dataset, examples, features
    return dataset


def evaluate(eval, test, args, model, tokenizer, output, prefix="", source_or_follow=False):
    if test:
        if source_or_follow:
            dataset, examples, features = load_and_cache_examples(args, tokenizer, train=False, evaluate=True, test=False,
                                                              output_examples=True, s_or_f=True)
        else:
            dataset, examples, features = load_and_cache_examples(args, tokenizer, train=False, evaluate=True, test=False,
                                                              output_examples=True, s_or_f=False)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].to("cpu").numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids, output)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1
        nb_eval_examples += inputs["input_ids"].size(0)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("%s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mr",
        default='1-1',
        type=str,
        required=True
    )
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        required=True,
        help="BoolQ jsonl"
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " ,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.server_ip and args.server_port:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    config.num_labels = 1
    model = model_class.from_pretrained(
        args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config
    )

    if args.local_rank == 0:
        torch.distributed.barrier() 

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    results = {}
    output_sources = []
    if args.do_eval and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            eval = False
            test = True
            result = evaluate(eval, test, args, model, tokenizer, output_sources, global_step, source_or_follow=True)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    output_follows = []
    if args.do_eval and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            tokenizer = tokenizer_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            eval = False
            test = True
            result = evaluate(eval, test, args, model, tokenizer, output_follows, global_step, source_or_follow=False)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    violations = []
    num = 0
    for source in output_sources:
        follow = output_follows[num]
        if source != follow:
            violations.append(source)
        num += 1
    if args.mr in ["1-1", "1-2", "1-3", "1-4"]:
        print("violation rate: ", 1-len(violations)/len(output_sources))
    elif args.mr in ["2-1", "2-2", "2-3"]:
        print("violation rate: ", len(violations)/len(output_sources))

    return results

if __name__ == "__main__":
    main()
