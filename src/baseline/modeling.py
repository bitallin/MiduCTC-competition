#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from src.baseline.ctc_vocab.config import VocabConf
from src.baseline.loss import LabelSmoothingLoss
from transformers.models.bert import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss

class ModelingCtcBert(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.tag_detect_projection_layer = torch.nn.Linear(
            config.hidden_size, VocabConf.detect_vocab_size)
        self.tag_label_projection_layer = torch.nn.Linear(
            config.hidden_size, VocabConf.correct_vocab_size)
        self.init_weights()
        self._detect_criterion = CrossEntropyLoss(ignore_index=-100)
        self._correct_criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=-100)

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        inputs['detect_labels'] = torch.zeros(size=(8, 56)).long()
        inputs['correct_labels'] = torch.zeros(size=(8, 56)).long()
        return inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        detect_labels=None,
        correct_labels=None
    ):

        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)[0]
        detect_outputs = self.tag_detect_projection_layer(hidden_states)
        correct_output = self.tag_label_projection_layer(hidden_states)

        loss = None
        if detect_labels is not None and correct_labels is not None:

            loss = self._detect_criterion(
                detect_outputs.view(-1, VocabConf.detect_vocab_size), detect_labels.view(-1)) + self._correct_criterion(
                correct_output.view(-1, VocabConf.correct_vocab_size), correct_labels.view(-1))
        elif detect_labels is not None:
            loss = self._detect_criterion(
                detect_outputs.view(-1, VocabConf.detect_vocab_size), detect_labels.view(-1))
        elif correct_labels is not None:
            loss = self._correct_criterion(
                correct_output.view(-1, VocabConf.correct_vocab_size), correct_labels.view(-1))

        return detect_outputs, correct_output, loss