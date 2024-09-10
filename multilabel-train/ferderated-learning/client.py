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
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
predicted_labels_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0
}
class DomainAdaptationClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader,source_dataloader,target_dataloader,device,trainning_params):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.device = device
        self.trainning_params = trainning_params 

    def get_parameters(self,config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Starting Trainning...")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        max_batches = min(len(source_dataloader), len(target_dataloader))
        loss_fn_sentiment_classifier = torch.nn.NLLLoss()
        loss_fn_domain_classifier = torch.nn.NLLLoss()
        for epoch_idx in range(1):
            source_iterator = iter(source_dataloader)
            target_iterator = iter(target_dataloader)
            for batch_idx in range(max_batches):
                # p = float(batch_idx + epoch_idx * max_batches) / (1 * max_batches)
                # grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
                # grl_lambda = torch.tensor(grl_lambda)
                optimizer.zero_grad()
                # Souce dataset training update
                input_ids, attention_mask, token_type_ids, labels = next(source_iterator)
                inputs = {
                    "input_ids": input_ids.squeeze(axis=1),
                    "attention_mask": attention_mask.squeeze(axis=1),
                    "token_type_ids" : token_type_ids.squeeze(axis=1),
                    "labels" : labels,
                    # "grl_lambda" : grl_lambda,
                }
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                print("Starting Trainning...")
                sentiment_pred, domain_pred = model(**inputs)
                loss_s_sentiment = loss_fn_sentiment_classifier(sentiment_pred, inputs["labels"])
                y_s_domain = torch.zeros(self.trainning_params["batch_size"], dtype=torch.long).to(device)
                loss_s_domain = loss_fn_domain_classifier(domain_pred, y_s_domain)
                # Target dataset training update
                input_ids, attention_mask, token_type_ids, labels = next(target_iterator)
                inputs = {
                    "input_ids": input_ids.squeeze(axis=1),
                    "attention_mask": attention_mask.squeeze(axis=1),
                    "token_type_ids" : token_type_ids.squeeze(axis=1),
                    "labels" : labels,
                    # "grl_lambda" : grl_lambda,
                }
                print("Starting Trainning...")
                for k, v in inputs.items():
                    inputs[k] = v.to(device)

                _, domain_pred = model(**inputs)

                # Note that we are not using the sentiment predictions here for updating the weights
                y_t_domain = torch.ones(input_ids.shape[0], dtype=torch.long).to(device)
                # print(domain_pred.shape, y_t_domain.shape)
                loss_t_domain = loss_fn_domain_classifier(domain_pred, y_t_domain)
                loss = loss_s_sentiment + loss_s_domain + loss_t_domain
                loss.backward()
                optimizer.step()
                print("Starting Trainning...")
                torch.cuda.empty_cache()
                if (batch_idx + 1) % 10 == 0 or batch_idx == max_batches - 1:
                    print(f"Batch {batch_idx + 1}/{max_batches} completed: {(batch_idx + 1) / max_batches * 100:.2f}%")
    #  def fit(self, parameters, config):
    #     self.set_parameters(parameters)
    #     print("Starting Trainning...")
    #     self.model.train()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # total_batches = len(self.train_loader)
        # for epoch in range(1):  # Perform a single epoch of training
        #     for batch_idx, batch in enumerate(self.train_loader):
        #         input_ids, attention_mask, token_type_ids, labels = [x.to(self.device) for x in batch]
        #         inputs = {
        #                 "input_ids": input_ids.squeeze(axis=1),
        #                 "attention_mask": attention_mask.squeeze(axis=1),
        #                 "token_type_ids" : token_type_ids.squeeze(axis=1),
        #                 "labels" : labels
        #             }
        #         for k, v in inputs.items():
        #             inputs[k] = v.to(device)
        #         # sentiment_pred, domain_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #         sentiment_pred, domain_pred = self.model(**inputs)
        #         loss = self.compute_loss(sentiment_pred, domain_pred, labels)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         torch.cuda.empty_cache()
        #         if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
        #             print(f"Batch {batch_idx + 1}/{total_batches} completed: {(batch_idx + 1) / total_batches * 100:.2f}%")
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
            for batch_idx, (input_ids, attention_mask, token_type_ids, labels) in enumerate(self.target_dataloader):
                inputs = {
                    "input_ids": input_ids.squeeze(axis=1).to(self.device),
                    "attention_mask": attention_mask.squeeze(axis=1).to(self.device),
                    "token_type_ids": token_type_ids.squeeze(axis=1).to(self.device),
                    "labels": labels.to(self.device),
                }
                sentiment_pred, _ = self.model(**inputs)
                _, predicted = torch.max(sentiment_pred, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return float(accuracy)
    # def evaluate(self, parameters, config):
    #     self.set_parameters(parameters)
    #     self.model.eval()
    #     correct = 0
    #     total = 0
    #     print("Starting evaluation...")
    #     total_batches = len(self.test_loader)
    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(self.test_loader):
    #             input_ids, attention_mask, token_type_ids, labels = [x.to(self.device) for x in batch]
    #             inputs = {
    #                     "input_ids": input_ids.squeeze(axis=1),
    #                     "attention_mask": attention_mask.squeeze(axis=1),
    #                     "token_type_ids" : token_type_ids.squeeze(axis=1),
    #                     "labels" : labels
    #                 }
    #             for k, v in inputs.items():
    #                 inputs[k] = v.to(device)
    #             # sentiment_pred, domain_pred = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    #             sentiment_pred, domain_pred = self.model(**inputs)
    #             _, predicted = torch.max(sentiment_pred, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #             torch.cuda.empty_cache()
    #             if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
    #                 print(f"Batch {batch_idx + 1}/{total_batches} completed: {(batch_idx + 1) / total_batches * 100:.2f}% accuracy: {correct/total * 100:.2f}")
    #     accuracy = correct / total
    #     return float(accuracy), len(self.test_loader.dataset),{"accuracy": accuracy}
    
#Model
device = torch.device("cuda")
model = DomainAdaptationModel()
tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('C:/Users/hl100/Downloads/size250k_1epoch_1_model.bin', map_location=device), strict=False)
model.to(device)
#Data
df_full = pd.read_csv('D:\Hoc\SecBert\SecBERT\multilabel-train\dataset_capec.csv')
df_full['text'] = df_full['text'].str.replace('/',' ')
df_train = df_full.groupby('label').head(20)

# df_train = df_full
df_train = df_train.dropna(subset=['label'])
label_counts = df_train['label'].value_counts()
X_train, X_test, Y_train, Y_test = train_test_split(df_train['text'], df_train['label'],test_size=0.3, stratify=df_train['label'], shuffle = True)
df_train = pd.concat([X_train, Y_train], axis=1)
df_test = pd.concat([X_test, Y_test], axis=1)
# Tokenize the loaded texts for training and testing
train_dataset = ReviewDataset(df_train)
test_dataset = ReviewDataset(df_test)

training_parameters = {
    "batch_size": 2
}
# DataLoader
source_dataset = ReviewDataset(df_train)
source_dataloader = DataLoader(dataset = source_dataset, batch_size = training_parameters["batch_size"], shuffle = True, num_workers = 2)
target_dataset = ReviewDataset(df_train)
target_dataloader = DataLoader(dataset = target_dataset, batch_size = training_parameters["batch_size"], shuffle = True, num_workers = 2)
train_loader = DataLoader(train_dataset, batch_size=training_parameters["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=training_parameters["batch_size"], shuffle=False) 

# Simulate clients
client = DomainAdaptationClient(model, train_loader, test_loader,source_dataloader,target_dataloader, device,training_parameters)
fl.client.start_numpy_client(
        server_address="localhost:8088",
        client =client
    )