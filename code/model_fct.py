import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchmetrics
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy,MultilabelAccuracy, MultilabelF1Score
from torchmetrics import Recall, Precision
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict

#from pytorch_lightning.metrics.sklearns import Accuracy

target_list = ['cyto', 'mito', 'nucleus','other', 'secreted']
#PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
EPOCHS = 2
BATCH_SIZE = 1
MAX_LENGTH = 1500

class ProteinSequenceDataset(Dataset):
    
    def __init__(self, data, tokenizer,target_list, max_len):
        
        self.data = data
        self.tokenizer = tokenizer
        self.target_list = target_list
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        
        single_row = self.data.iloc[item]
        sequence = single_row['Sequences']
        target = single_row[target_list]
        
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.float)
        }


class ProteinDataModule(pl.LightningDataModule):
    
    def __init__(self, train_df, test_df,val_df,blind_df,  tokenizer, target_list, batch_size=32, max_len=1500):
        super().__init__()
        
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.blind_df = blind_df
        self.target_list = target_list
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
    
    def setup(self, stage=None):
        self.train_dataset = ProteinSequenceDataset(self.train_df, self.tokenizer,self.target_list, self.max_len)
        self.test_dataset = ProteinSequenceDataset(self.test_df, self.tokenizer,self.target_list, self.max_len)
        self.val_dataset = ProteinSequenceDataset(self.val_df, self.tokenizer,self.target_list, self.max_len)
        self.blind_dataset = ProteinSequenceDataset(self.blind_df, self.tokenizer,self.target_list, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=4)    
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.blind_dataset,batch_size=self.batch_size,num_workers=4)



class ProteinClassifier(pl.LightningModule):
    def __init__(self, n_classes: int,target_list, steps_per_epoch=None, n_epochs=None):
        super().__init__()

        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        #check for the parameters of the linear layer
        self.classifier = nn.Sequential(#nn.Dropout(p=0.2),
                                        nn.Linear(self.bert.config.hidden_size, n_classes),
                                        nn.Tanh()
                                        #nn.Softmax()
        )
        
        self.target_list = target_list
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.metric_acc = MultilabelAccuracy(num_labels=5)
        self.metric_f1 = MultilabelF1Score(num_labels=5)
        self.metric_precision = Precision(task="multilabel", average='macro', num_labels=5)
        self.metric_recall = Recall("multilabel", average='macro', num_labels=5)
        self.save_hyperparameters()
        
    def forward(self, input_ids, attention_mask, targets=None):
        
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        
        loss = 0
        if targets is not None:
            loss = self.criterion(output, targets)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        loss, outputs = self(input_ids, attention_mask, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        #training_acc = self.metric_acc(outputs, targets)
        self.log("training_acc", self.metric_acc(outputs, targets), on_step=False, on_epoch=True)
        
        #training_f1 = self.metric_f1(outputs, targets)
        self.log('training_f1', self.metric_f1(outputs, targets), on_step=False, on_epoch=True)
        
        #training_recall = self.metric_recall(outputs, targets)
        self.log('training_recall', self.metric_recall(outputs, targets), on_step=False, on_epoch=True)
        
        #training_precision = self.metric_precision(outputs, targets)
        self.log('training_precision', self.metric_precision(outputs, targets), on_step=False, on_epoch=True)
        
        return OrderedDict({
            "loss": loss,
            "predictions": outputs,
            "targets": targets,
            #"training_acc": training_acc
        })
    

    def validation_step(self, batch, batch_idx):
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        loss, outputs = self(input_ids, attention_mask, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        #val_acc = self.metric_acc(outputs, targets)
        self.log('val_acc', self.metric_acc(outputs, targets), on_step=False, on_epoch=True)
        
        #val_f1 = self.metric_f1(outputs, targets)
        self.log('val_f1', self.metric_f1(outputs, targets), on_step=False, on_epoch=True)
        
        #val_recall = self.metric_recall(outputs, targets)
        self.log('val_recall', self.metric_recall(outputs, targets), on_step=False, on_epoch=True)
        
        #val_precision = self.metric_precision(outputs, targets)
        self.log('val_precision', self.metric_precision(outputs, targets), on_step=False, on_epoch=True)

        return OrderedDict({
            "val_loss": loss,
            "predictions": outputs,
            "targets": targets,
            #"val_acc": val_acc,
            #'val_f1':val_f1,
            #'val_recall':val_recall,
            #'val_precision':val_precision
        })
    
    #def validation_epoch_end(self, outputs):

        
        #val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        #val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        #val_f1_mean = torch.stack([x['val_f1'] for x in outputs]).mean()
        #val_precision_mean = torch.stack([x['val_recall'] for x in outputs]).mean()
        #val_recall_mean = torch.stack([x['val_precision'] for x in outputs]).mean()

       
        #tqdm_dict = {"val_loss": val_loss_mean, 
                     #"val_acc": val_acc_mean
                     #'val_f1':val_f1,
                     #'val_recall':val_recall,
                     #'val_precision':val_precision
                    #}
        
        #result = {
            #"progress_bar": tqdm_dict,
            ##"log": tqdm_dict,
            #"val_loss": val_loss_mean,
        #}
        #return result
        
    def test_step(self, batch, batch_idx):
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        
        loss, outputs = self(input_ids, attention_mask, targets)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        
        #test_acc = self.metric_acc(outputs, targets)
        self.log("test_acc", self.metric_acc(outputs, targets), on_step=False, on_epoch=True)
        
        #test_f1 = self.metric_f1(outputs, targets)
        self.log('test_f1', self.metric_f1(outputs, targets), on_step=False, on_epoch=True)
        
        #test_recall = self.metric_recall(outputs, targets)
        self.log('test_recall', self.metric_recall(outputs, targets), on_step=False, on_epoch=True)
        
        #test_precision = self.metric_precision(outputs, targets)
        self.log('test_precision', self.metric_precision(outputs, targets), on_step=False, on_epoch=True)

        return OrderedDict({
            "test_loss": loss,
            "predictions": outputs,
            "targets": targets,
            #"test_acc": test_acc,
            #'test_f1':test_f1,
            #'test_recall':test_recall,
            #'test_precision':test_precision
        })
    
    #def testing_epoch_end(self, outputs):

        
        #test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        #test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()
        #test_f1_mean = torch.stack([x['test_f1'] for x in outputs]).mean()
        #test_precision_mean = torch.stack([x['test_recall'] for x in outputs]).mean()
        #test_recall_mean = torch.stack([x['test_precision'] for x in outputs]).mean()

       
        #tqdm_dict = {"test_loss": test_loss_mean, 
                     #"test_acc": test_acc_mean
                     #'test_f1':test_f1,
                     #'test_recall':test_recall,
                     #'test_precision':test_precision
                    #}
        
        #result = {
            #"progress_bar": tqdm_dict,
            #"log": tqdm_dict,
            #"test_loss": test_loss_mean,
        #}
        #return result
    
    def configure_optimizers(self):
        
        parameters = [
            {"params": self.classifier.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": 5e-06,
            },
        ]
        optimizer = AdamW(parameters, lr=2e-5)
        #warmup_steps = self.steps_per_epoch // 3
        #total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
        #scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], []
    
    def predict_step(self, batch,batch_idx):
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        return self(input_ids, attention_mask)

        
        
