#!pip install flwr
import flwr as fl
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
from centralized import DomainAdaptationModel,ReviewDataset
from sklearn.model_selection import train_test_split

class DomainAdaptationClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(1):  # Perform a single epoch of training
            for batch in self.train_loader:
                input_ids, attention_mask, token_type_ids, labels = [x.to(self.device) for x in batch]
                sentiment_pred, domain_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = self.compute_loss(sentiment_pred, domain_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def compute_loss(self, sentiment_pred, domain_pred, labels):
        sentiment_loss = torch.nn.CrossEntropyLoss()(sentiment_pred, labels)
        domain_loss = torch.nn.CrossEntropyLoss()(domain_pred, labels)  # Assuming domain labels are in labels
        return sentiment_loss + domain_loss

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask, token_type_ids, labels = [x.to(self.device) for x in batch]
                sentiment_pred, domain_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                _, predicted = torch.max(sentiment_pred, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return float(accuracy), len(self.test_loader.dataset), {}
    

#Model
model = DomainAdaptationModel()
tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#Data
df_full = pd.read_csv('D:\Hoc\SecBert\SecBERT\multilabel-train\dataset_capec.csv')
df_full['text'] = df_full['text'].str.replace('/',' ')
df_train = df_full.groupby('label').head(2000)
# df_train = df_full
df_train = df_train.dropna(subset=['label'])
label_counts = df_train['label'].value_counts()
train_df, test_df = train_test_split(df_train, test_size=0.2, random_state=42)
train_texts = train_df['text'].values
train_labels = train_df['label'].values
test_texts = test_df['text'].values
test_labels = test_df['label'].values
# Tokenize the loaded texts for training and testing
train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
test_dataset = ReviewDataset(test_texts, test_labels, tokenizer)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False) 

# Simulate clients
client = DomainAdaptationClient(model, train_loader, test_loader, device)
fl.client.start_numpy_client(server_address="localhost:8080", client=client)