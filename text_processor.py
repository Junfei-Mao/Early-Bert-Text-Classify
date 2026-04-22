# -*- coding: utf-8 -*-
"""Data processing helpers for text classification."""

import copy
import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

LABELS_FILENAME = "label_config.json"
TASK_NAME_ALIASES = {
    "text_classify": "text_classify",
    "text_similar": "text_classify",
}


def normalize_task_name(task_name):
    task_name = task_name.lower()
    if task_name not in TASK_NAME_ALIASES:
        raise ValueError(f"Unsupported task name: {task_name}")
    return TASK_NAME_ALIASES[task_name]


def build_text(title, content):
    title = (title or "").strip()
    content = (content or "").strip()
    if title and content:
        return f"{title}\n{content}"
    return title or content


def read_jsonl_records(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8-sig") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_labels(labels_path=None, data_dir=None):
    if labels_path:
        labels_path = Path(labels_path)
        if labels_path.suffix.lower() == ".json":
            with open(labels_path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if isinstance(payload, dict):
                labels = payload.get("labels", [])
            elif isinstance(payload, list):
                labels = payload
            else:
                raise ValueError(f"Unsupported label file format: {labels_path}")
        else:
            with open(labels_path, "r", encoding="utf-8") as file_obj:
                labels = [line.strip() for line in file_obj if line.strip()]
        if not labels:
            raise ValueError(f"No labels found in {labels_path}")
        return labels

    if not data_dir:
        raise ValueError("Either labels_path or data_dir must be provided.")

    labels = set()
    for file_name in ("train.tsv", "dev.tsv", "test.tsv"):
        data_file = Path(data_dir) / file_name
        if not data_file.exists():
            continue
        for record in read_jsonl_records(data_file):
            label = record.get("label")
            if label:
                labels.add(label)

    if not labels:
        raise ValueError(
            "No labels found. Please provide --labels or ensure data files contain a `label` field."
        )
    return sorted(labels)


def save_labels(output_dir, labels):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label_path = output_path / LABELS_FILENAME
    with open(label_path, "w", encoding="utf-8") as file_obj:
        json.dump({"labels": labels}, file_obj, ensure_ascii=False, indent=2)
    return label_path


def load_labels_from_model_dir(model_dir):
    return load_labels(labels_path=Path(model_dir) / LABELS_FILENAME)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8-sig") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    lines.append(line)
        return lines


class InputExample(object):
    """A single training/test example for sequence classification."""

    def __init__(self, guid, text_a, text_b=None, classify_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.classify_label = classify_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn(batch):
    """Trim each batch to the max valid sequence length of that batch."""
    all_a_input_ids, all_a_input_mask, all_a_segment_ids, all_label_ids = map(torch.stack, zip(*batch))
    valid_lengths = all_a_input_mask.sum(dim=1)
    max_len = int(valid_lengths.max().item()) if valid_lengths.numel() else all_a_input_ids.size(1)
    max_len = max(max_len, 1)
    all_a_input_ids = all_a_input_ids[:, :max_len]
    all_a_input_mask = all_a_input_mask[:, :max_len]
    all_a_segment_ids = all_a_segment_ids[:, :max_len]
    return all_a_input_ids, all_a_input_mask, all_a_segment_ids, all_label_ids


def convert_examples_to_features(
    examples,
    label2id,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of InputFeature objects."""
    del cls_token_at_end
    del pad_on_left

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        if example.classify_label not in label2id:
            raise KeyError(f"Unknown label `{example.classify_label}` found in example {example.guid}")

        label_id = label2id[example.classify_label]
        tokens_a = tokenizer.tokenize(example.text_a)
        special_tokens_count = 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[: max_seq_length - special_tokens_count]

        tokens_a = [cls_token] + tokens_a + [sep_token]
        a_segment_ids = [sequence_a_segment_id] * len(tokens_a)
        a_input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
        a_input_mask = [1 if mask_padding_with_zero else 0] * len(a_input_ids)

        a_padding_length = max_seq_length - len(a_input_ids)
        a_input_ids += [pad_token] * a_padding_length
        a_input_mask += [0] * a_padding_length
        a_segment_ids += [pad_token_segment_id] * a_padding_length

        assert len(a_input_ids) == max_seq_length
        assert len(a_input_mask) == max_seq_length
        assert len(a_segment_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens_a: %s", " ".join([str(x) for x in tokens_a]))
            logger.info("a_input_ids: %s", " ".join([str(x) for x in a_input_ids]))
            logger.info("a_input_mask: %s", " ".join([str(x) for x in a_input_mask]))
            logger.info("a_segment_ids: %s", " ".join([str(x) for x in a_segment_ids]))

        features.append(
            InputFeature(
                input_ids=a_input_ids,
                input_mask=a_input_mask,
                segment_ids=a_segment_ids,
                label_id=label_id,
            )
        )
    return features


def _create_examples(lines, set_type):
    examples = []
    for index, line in enumerate(lines):
        guid = f"{set_type}-{index}"
        json_data = json.loads(line) if isinstance(line, str) else line
        text = build_text(json_data.get("title", ""), json_data.get("content", ""))
        examples.append(
            InputExample(
                guid=guid,
                text_a=text,
                classify_label=json_data.get("label"),
            )
        )
    return examples


class TextClassifyProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return _create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return _create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return _create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        raise NotImplementedError("Use load_labels(...) to load labels dynamically.")

    def get_predict_examples(self, records):
        return _create_examples(records, "predict")


cls_processors = {
    "text_classify": TextClassifyProcessor,
    "text_similar": TextClassifyProcessor,
}
