# -*- coding: utf-8 -*-
"""Inference entrypoint for the trained text classification model."""

import argparse
import json

import torch
from transformers import BertTokenizer

from bert_for_cls import BertForTextClassify
from text_processor import build_text, load_labels_from_model_dir


class TextClassifierPredictor:
    def __init__(self, model_dir="./output", device=None, max_length=512):
        self.model_dir = model_dir
        self.label_list = load_labels_from_model_dir(model_dir)
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.max_length = max_length

        self.model = BertForTextClassify.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, title="", content=""):
        text = build_text(title, content)
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        logits = self.model(**encoded)[0]
        probabilities = torch.softmax(logits, dim=1).cpu()[0]
        label_id = int(torch.argmax(probabilities).item())
        return {
            "label": self.label_list[label_id],
            "label_id": label_id,
            "confidence": round(float(probabilities[label_id]), 6),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained BERT text classifier.")
    parser.add_argument("--model_dir", default="./output", help="Directory containing the exported model.")
    parser.add_argument("--title", default="", help="Optional title field.")
    parser.add_argument("--content", default="", help="Content to classify.")
    parser.add_argument("--device", default=None, help="Device name, for example cpu or cuda:0.")
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode.")
    return parser.parse_args()


def interactive_loop(predictor):
    while True:
        print("\nInput content (empty line to exit):")
        content = input().strip()
        if not content:
            break
        print("Input title (optional):")
        title = input().strip()
        result = predictor.predict(title=title, content=content)
        print(json.dumps(result, ensure_ascii=False))


def main():
    args = parse_args()
    predictor = TextClassifierPredictor(
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )

    if args.interactive:
        interactive_loop(predictor)
        return

    if not args.content:
        raise ValueError("Please provide --content, or use --interactive.")

    result = predictor.predict(title=args.title, content=args.content)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
