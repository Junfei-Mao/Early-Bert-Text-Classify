# -*- coding:utf-8 -*-
"""Model definition for BERT-based text classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss


class BertForTextClassify(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTextClassify, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.lstm_hidden_size = 256
        self.bert = BertModel(config)  # .from_pretrained(config.model_name_or_path)
        self.bilstm_layer = nn.LSTM(input_size=config.hidden_size, hidden_size=self.lstm_hidden_size,
                                    batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(p=0.2)
        self.weight_W = nn.Parameter(torch.Tensor(2 * self.lstm_hidden_size, 2 * self.lstm_hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * self.lstm_hidden_size, 1))
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(self.lstm_hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        sequence_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        sequence_output = self.dropout(sequence_output)
        lstm_state, lstm_hidden = self.bilstm_layer(sequence_output)
        u = torch.tanh(torch.matmul(lstm_state, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = lstm_state * att_score
        lstm_state = self.lstm_dropout(lstm_state)
        out = torch.sum(scored_x, dim=1)
        logits = self.classifier(out)
        outputs = (logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
