import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import newaxis
import math
import os
import pandas as pd
import torch.nn as nn
from scipy.stats import chi2
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


config = {
    "num_labels": 9,
    "hidden_dropout_prob": 0.15,
    "hidden_size": 768,
    "max_length": 512,
}    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DomainAdaptationModel(nn.Module):

    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        num_labels = config["num_labels"]
        self.bert = AutoModel.from_pretrained('jackaduma/SecBERT')
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], num_labels),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          labels=None,
          grl_lambda = 1.0,
          ):

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

#         pooled_output = outputs[1] # For bert-base-uncase
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)


        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)

        sentiment_pred = self.sentiment_classifier(pooled_output)
        domain_pred = self.domain_classifier(reversed_pooled_output)

        return sentiment_pred.to(device), domain_pred.to(device)


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
attack_dict = {
    '000 - Normal': 0,
    '126 - Path Traversal': 1,
    '242 - Code Injection': 2,
    '153 - Input Data Manipulation': 3,
    '310 - Scanning for Vulnerable Software': 4,
    '194 - Fake the Source of Data': 5,
    '34 - HTTP Response Splitting': 6,
    '66 - SQL Injection':7,
    '272 - Protocol Manipulation':8
            }
class ReviewDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('jackaduma/secBERT')
    def __getitem__(self, index):
        review = self.df.iloc[index]["text"]
        attack = self.df.iloc[index]["label"]

        label = attack_dict[attack]
        encoded_input = self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length = 512,
                padding="max_length",
                return_overflowing_tokens=True,
                truncation = True,
            )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            # print("Attention! you are cropping tokens")
            pass

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None



        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(label),
        }

        return data_input["input_ids"], data_input["attention_mask"], data_input["token_type_ids"], data_input["label"]



    def __len__(self):
        return self.df.shape[0]