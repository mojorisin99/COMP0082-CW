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
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy,MulticlassAccuracy, MulticlassF1Score
from torchmetrics import Recall, Precision, ConfusionMatrix
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


target_list = ['cyto', 'mito', 'nucleus','other', 'secreted']
num_samples = [3004,1604,1299,3014,2002]

#PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
EPOCHS = 3
BATCH_SIZE = 1
MAX_LENGTH = 1500

class ProteinSequenceDataset(Dataset):
    """
    This class instantiates a dataset that contains all the sequences of the protein dataset
    """
    def __init__(self, data, tokenizer,target_list, max_len):
        """
        Constructor method that initializes the class instance with the given arguments.

        Args:
            data (pandas DataFrame): DataFrame containing protein sequence data.
            tokenizer (Tokenizer): The tokenizer to use to encode the protein sequences.
            target_list (list): List of column names in `data` that contains the target values for each sequence.
            max_len (int): The maximum length of the protein sequence to use for encoding.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.target_list = target_list
        self.max_len = max_len

    def __len__(self):
        """
        Returns the length of the dataset.

        Output:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Returns the data and target values for a single item in the dataset.

        Args:
            item (int): Index of the item to retrieve from the dataset.

        Output:
            dict: A dictionary that contains the following keys:
                'protein_sequence': str, the protein sequence string.
                'input_ids': torch.Tensor, the encoded input_ids tensor.
                'attention_mask': torch.Tensor, the encoded attention_mask tensor.
                'targets': torch.Tensor, the target tensor.
        """
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
    """
    This class creates a Pytorch Lightning data module using the ProteinSequenceDataset for the
    four different datasets (train, val, test and predictions)
    """
    def __init__(self, train_df, test_df,val_df,blind_df,  tokenizer, target_list, batch_size=32, max_len=1500):
        """
        Creates an instance of the ProteinDataModule and sets its arguments

        Args:
            - train_df (pandas.DataFrame): training set of protein sequences and target labels
            - test_df (pandas.DataFrame): test set of protein sequences and target labels
            - val_df (pandas.DataFrame): validation set of protein sequences and target labels
            - blind_df (pandas.DataFrame): blind set of protein sequences
            - tokenizer (Tokenizer): Tokenizer for encoding the protein sequences
            - target_list (list): List of target labels
            - batch_size (int): Batch size for dataloading
            - max_len (int): Max length of protein sequences for tokenization
        """
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
        """
        Setup the dataset using the ProteinSequenceDataset class
        """
        self.train_dataset = ProteinSequenceDataset(self.train_df, self.tokenizer,self.target_list, self.max_len)
        self.test_dataset = ProteinSequenceDataset(self.test_df, self.tokenizer,self.target_list, self.max_len)
        self.val_dataset = ProteinSequenceDataset(self.val_df, self.tokenizer,self.target_list, self.max_len)
        self.blind_dataset = ProteinSequenceDataset(self.blind_df, self.tokenizer,self.target_list, self.max_len)

    #The four functions below create the four needed dataloaders for the classifier
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.blind_dataset,batch_size=self.batch_size,num_workers=4)



class ProteinClassifier(pl.LightningModule):
    """
    This class creates a Pytorch Lightning Classifier using the ProteinDatModule for the
    four different dataloaders (train, val, test, predict)
    """
    def __init__(self, n_classes: int,target_list, steps_per_epoch=None, n_epochs=None):
        """
        This function initializes the ProtBert Classifier module
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, n_classes),
                                        nn.Softmax()
        )
        self.n_classes = n_classes
        self.target_list = target_list
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

        self.val_targets = []
        self.val_preds = []

        self.test_targets = []
        self.test_preds = []


        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, targets=None):
        """
        Computes the forward pass of the model.
        """
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)


        loss = 0
        if targets is not None:
            loss = self.criterion(output, targets)
        return loss, output

    def training_step(self, batch, batch_idx):
        """
        Computes the training step
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]
        loss, outputs = self(input_ids, attention_mask, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        return OrderedDict({
            "loss": loss,
            "predictions": outputs,
            "targets": targets
        })


    def validation_step(self, batch, batch_idx):
        """
        Computes the validation step
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        loss, outputs = self(input_ids, attention_mask, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        outputs = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)

        self.val_targets.append(targets.detach().cpu().numpy())
        self.val_preds.append(outputs.detach().cpu().numpy())

        return OrderedDict({
            "val_loss": loss,
            "predictions": outputs,
            "targets": targets
        })


    def test_step(self, batch, batch_idx):
        """
        Computes the test step
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        loss, outputs = self(input_ids, attention_mask, targets)
        #print(outputs)
        outputs = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)

        self.test_targets.append(targets.detach().cpu().numpy())
        self.test_preds.append(outputs.detach().cpu().numpy())


        return OrderedDict({
            "test_loss": loss,
            "predictions": outputs,
            "targets": targets
        })

    def validation_epoch_end(self, outputs):
        """
        Function executed at the end of the validation steps
        """
        val_targets = np.concatenate(self.val_targets)
        val_preds = np.concatenate(self.val_preds)

        df = pd.DataFrame({'target': val_targets, 'prediction': val_preds})

        # compute global metrics
        accuracy = accuracy_score(df['target'], df['prediction'])
        precision = precision_score(df['target'], df['prediction'], average='macro')
        recall = recall_score(df['target'], df['prediction'], average='macro')
        f1 = f1_score(df['target'], df['prediction'], average='macro')

        print(f"the accuracy is {accuracy:.2f}")
        print(f"the precision is {precision:.2f}")
        print(f"the recall is {recall:.2f}")
        print(f"the f1 is {f1:.2f}")

        cm = confusion_matrix(val_targets, val_preds)

        # compute total number of samples for each class
        total_per_class = np.sum(cm, axis=1)

        # compute number of correctly classified samples for each class
        correct_per_class = np.diagonal(cm)

        # compute precision, recall, and f1 score for each class
        p, r, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average=None)

        # compute accuracy for each class
        accuracy_per_class = np.divide(correct_per_class, total_per_class, where=total_per_class!=0)

        # create a dataframe to hold the results
        df_class = pd.DataFrame({
            'precision': p,
            'recall': r,
            'f1': f1,
            'accuracy': accuracy_per_class,
            'num_samples': total_per_class
        })

        print(df_class)
        print(cm)

    def test_epoch_end(self, outputs):
        """
        Function executed at the end of the testing steps
        """
        test_targets = np.concatenate(self.test_targets)
        test_preds = np.concatenate(self.test_preds)

        df = pd.DataFrame({'target': test_targets, 'prediction': test_preds})

        # compute global metrics
        accuracy = accuracy_score(df['target'], df['prediction'])
        precision = precision_score(df['target'], df['prediction'], average='macro')
        recall = recall_score(df['target'], df['prediction'], average='macro')
        f1 = f1_score(df['target'], df['prediction'], average='macro')

        print(f"the accuracy is {accuracy:.2f}")
        print(f"the precision is {precision:.2f}")
        print(f"the recall is {recall:.2f}")
        print(f"the f1 is {f1:.2f}")

        cm = confusion_matrix(test_targets, test_preds)

        # compute total number of samples for each class
        total_per_class = np.sum(cm, axis=1)

        # compute number of correctly classified samples for each class
        correct_per_class = np.diagonal(cm)

        # compute precision, recall, and f1 score for each class
        p, r, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average=None)

        # compute accuracy for each class
        accuracy_per_class = np.divide(correct_per_class, total_per_class, where=total_per_class!=0)

        # create a dataframe to hold the results
        df_class = pd.DataFrame({
            'precision': p,
            'recall': r,
            'f1': f1,
            'accuracy': accuracy_per_class,
            'num_samples': total_per_class
        })

        print(df_class)
        print(cm)

    def configure_optimizers(self):
        """
        Function that configures the Adam Optimiser for the model
        """
        parameters = [
            {"params": self.classifier.parameters()},
            {
                "params": self.bert.parameters(),
                "lr": 5e-06,
            },
        ]
        optimizer = AdamW(parameters, lr=2e-5)
        return [optimizer], []

    def predict_step(self, batch,batch_idx):
        """
        Function that computes the prediction step for the model
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        return self(input_ids, attention_mask)
