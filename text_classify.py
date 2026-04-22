# -*- coding: utf-8 -*-
"""Train and evaluate a BERT-based text classification model."""

import argparse
import glob
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, AutoTokenizer, BertConfig, WEIGHTS_NAME, get_linear_schedule_with_warmup

from bert_for_cls import BertForTextClassify
from progressbar import ProgressBar
from text_processor import (
    LABELS_FILENAME,
    cls_processors as processors,
    collate_fn,
    convert_examples_to_features,
    load_labels,
    normalize_task_name,
    save_labels,
)
from tools.common import init_logger, logger, seed_everything

warnings.filterwarnings("ignore")

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTextClassify, AutoTokenizer),
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_label_list(args):
    candidate_paths = []
    if args.labels:
        candidate_paths.append(Path(args.labels))

    for path_str in (args.output_dir, args.model_name_or_path):
        if not path_str:
            continue
        candidate_paths.append(Path(path_str) / LABELS_FILENAME)

    for candidate in candidate_paths:
        if candidate.exists():
            return load_labels(labels_path=candidate)

    return load_labels(data_dir=args.data_dir)


def build_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    head_modules = ["bilstm_layer", "classifier", "weight_W", "weight_proj"]
    all_params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for name, p in all_params if not any(tag in name for tag in no_decay + head_modules)],
            "weight_decay": 1e-2,
            "lr": args.learning_rate,
        },
        {
            "params": [
                p
                for name, p in all_params
                if not any(tag in name for tag in head_modules) and any(tag in name for tag in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate,
        },
        {
            "params": [
                p
                for name, p in all_params
                if any(tag in name for tag in head_modules) and any(tag in name for tag in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.classifier_learning_rate,
        },
        {
            "params": [
                p
                for name, p in all_params
                if any(tag in name for tag in head_modules) and not any(tag in name for tag in no_decay)
            ],
            "weight_decay": 1e-2,
            "lr": args.classifier_learning_rate,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)


def train(args, train_dataset, model, tokenizer):
    """Train the model."""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // max(1, len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = build_optimizer(model, args)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )

    optimizer_state = Path(args.model_name_or_path) / "optimizer.pt"
    scheduler_state = Path(args.model_name_or_path) / "scheduler.pt"
    if optimizer_state.is_file() and scheduler_state.is_file():
        optimizer.load_state_dict(torch.load(optimizer_state, map_location="cpu"))
        scheduler.load_state_dict(torch.load(scheduler_state, map_location="cpu"))

    if args.fp16:
        try:
            from apex import amp
        except ImportError as exc:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            ) from exc
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        steps_per_epoch = max(1, len(train_dataloader) // args.gradient_accumulation_steps)
        epochs_trained = global_step // steps_per_epoch
        steps_trained_in_current_epoch = global_step % steps_per_epoch
        logger.info("  Continuing training from checkpoint")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss = 0.0
    model.zero_grad()
    seed_everything(args.seed)
    max_acc = 0.0

    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc=f"Training epoch {epoch + 1}")
        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            pbar(step, {"loss": loss.item()})
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps != 0:
                continue

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scheduler.step()
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(" ")
                if args.local_rank == -1:
                    eval_loss, acc, report = evaluate(args, model, tokenizer)
                    print(f"Epoch: {epoch}, eval loss: {eval_loss}, eval acc: {acc}")
                    if acc > max_acc:
                        max_acc = acc
                        print("Precision, Recall and F1-Score...")
                        print(report)

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                output_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, output_dir / "training_args.bin")
                torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
                torch.save(scheduler.state_dict(), output_dir / "scheduler.pt")
                logger.info("Saving model checkpoint to %s", output_dir)

        print(" ")
        if args.device.type == "cuda":
            torch.cuda.empty_cache()

    eval_loss, acc, report = evaluate(args, model, tokenizer)
    print(f"result! eval loss: {eval_loss}, eval acc: {acc}")
    if acc > max_acc:
        print("Precision, Recall and F1-Score...")
        print(report)

    average_loss = tr_loss / max(global_step, 1)
    return global_step, average_loss


def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = Path(args.output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="dev")
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
    )

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    for step, batch in enumerate(eval_dataloader):
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

        labels = batch[3].cpu().numpy()
        predictions = torch.max(logits, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predictions)

        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)

    eval_loss = eval_loss / max(nb_eval_steps, 1)
    acc = accuracy_score(labels_all, predict_all)
    report = classification_report(
        labels_all,
        predict_all,
        labels=list(range(len(args.label_list))),
        target_names=args.label_list,
        zero_division=0,
    )
    return eval_loss, acc, report


def load_and_cache_examples(args, task, tokenizer, data_type="train"):
    is_evaluation = data_type != "train"
    if args.local_rank not in [-1, 0] and not is_evaluation:
        torch.distributed.barrier()

    processor = processors[task]()
    model_name = Path(args.model_name_or_path.rstrip("/")).name or "model"
    max_seq_length = args.train_max_seq_length if data_type == "train" else args.eval_max_seq_length
    cached_features_file = Path(args.data_dir) / f"cached_span-{data_type}_{model_name}_{max_seq_length}_{task}"

    if cached_features_file.exists() and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, map_location="cpu")
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label2id=args.label2id,
            max_seq_length=max_seq_length,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            pad_on_left=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not is_evaluation:
        torch.distributed.barrier()

    all_a_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_a_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_a_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_a_input_ids, all_a_input_mask, all_a_segment_ids, all_label_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a BERT text classifier.")

    parser.add_argument(
        "--task_name",
        default="text_classify",
        type=str,
        help="Task name. Supported aliases: " + ", ".join(sorted(processors.keys())),
    )
    parser.add_argument("--data_dir", default="./data", type=str, help="Directory containing train/dev/test TSV files.")
    parser.add_argument("--model_type", default="bert", type=str, help="Model type.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-chinese",
        type=str,
        help="Path or Hugging Face model id for the pretrained checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Directory where checkpoints and training artifacts will be written.",
    )

    parser.add_argument("--loss_type", default="ce", type=str, choices=["ce"])
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Optional label file (.json or .txt). If omitted, labels are inferred from the dataset.",
    )
    parser.add_argument("--config_name", default="", type=str, help="Optional config name or path.")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Optional tokenizer name or path.")
    parser.add_argument("--cache_dir", default="", type=str, help="Optional transformers cache directory.")
    parser.add_argument("--train_max_seq_length", default=512, type=int)
    parser.add_argument("--eval_max_seq_length", default=512, type=int)
    parser.add_argument("--do_train", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--do_eval", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--do_predict", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--classifier_learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-7, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--logging_steps", type=int, default=600)
    parser.add_argument("--save_steps", type=int, default=600)
    parser.add_argument("--eval_all_checkpoints", action="store_true")
    parser.add_argument("--predict_checkpoints", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", type=str, default="O1")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--server_ip", type=str, default="")
    parser.add_argument("--server_port", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logger(log_file=output_dir / f"{args.task_name}.log")

    if output_dir.exists() and any(output_dir.iterdir()) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir true to overwrite it."
        )

    if args.server_ip and args.server_port:
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    seed_everything(args.seed)
    args.task_name = normalize_task_name(args.task_name)
    if args.task_name not in processors:
        raise ValueError(f"Task not found: {args.task_name}")

    label_list = get_label_list(args)
    args.label_list = label_list
    args.id2label = {index: label for index, label in enumerate(label_list)}
    args.label2id = {label: index for index, label in enumerate(label_list)}
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        loss_type=args.loss_type,
        cache_dir=args.cache_dir if args.cache_dir else None,
        soft_label=True,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    if args.local_rank in [-1, 0]:
        save_labels(args.output_dir, label_list)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, output_dir / "training_args.bin")

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = [
                os.path.dirname(path) for path in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            ]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            prefix = checkpoint.split("/")[-1] if "checkpoint" in checkpoint else ""
            eval_loss, acc, report = evaluate(args, model, tokenizer, prefix=prefix)
            results[prefix or "best"] = {"eval_loss": eval_loss, "eval_acc": acc, "report": report}

        output_eval_file = output_dir / "eval_results.txt"
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            for key in sorted(results.keys()):
                writer.write(f"{key} = {results[key]}\n")

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = [
                os.path.dirname(path) for path in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            ]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
            checkpoints = [path for path in checkpoints if path.split("-")[-1] == str(args.predict_checkpoints)]
        logger.info("Predict checkpoints discovered: %s", checkpoints)


if __name__ == "__main__":
    main()
