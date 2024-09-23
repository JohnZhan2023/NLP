'''trainer module for training the model'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from classifier import classifier
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

class Trainer():
    def __init__(self, model,TEXT, LABEL, batch_size, lr, epochs):
        self.model = model
        self.encoder_type = model.encoder_type
        self.embedding_type = model.embedding_type
        self.attention = model.backbone.attention
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max= epochs//4)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_iter, valid_iter, test_iter):

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loss = 0
            for batch in train_iter:
                x = batch.text
                y = batch.label
                x = x.to(self.device)
                y = y.to(self.device)
                y = y - 1
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_iter)
            if epoch % 4 == 0:
                valid_loss, valid_acc = self.evaluate(valid_iter)
                print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Acc: {:.4f}'.format(epoch, train_loss,  valid_loss, valid_acc))
                save_path = f'ckpt/{self.encoder_type}_{self.embedding_type}_{self.attention}/{epoch}.pt'

                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                torch.save(self.model, save_path)
                if valid_acc > 0.3:
                    self.test(test_iter)
                #wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'valid_acc': valid_acc})

            #else:
                #print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, train_loss))
                #wandb.log({'train_loss': train_loss})
            self.scheduler.step()
            
    def evaluate(self, valid_iter):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for batch in valid_iter:
                x = batch.text
                y = batch.label
                y = y - 1
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss += self.criterion(output, y).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
        loss /= len(valid_iter)
        acc = correct / len(valid_iter.dataset)

        return loss, acc
    def test(self, test_iter):
    # test the model and calculate the accuracy
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_iter:
                x = batch.text
                y = batch.label
                x = x.to(self.device)
                y = y.to(self.device)
                y = y - 1
                output = self.model(x)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
        acc = correct / len(test_iter.dataset)
        print('Test Accuracy: {:.4f}'.format(acc))
        #wandb.log({'test_acc': acc})

if __name__ == '__main__':
    os.environ["WANDB_DISABLED"] = "true"    # disable wandb for this script
    # set the random seed for reproducibility
    torch.manual_seed(0)
    TEXT = torchtext.data.Field(lower=True,fix_length=200,batch_first=True)
    LABEL = torchtext.data.Field(sequential=False)
    train,valid,test = torchtext.datasets.SST.splits(TEXT,LABEL, fine_grained=True)
    # TEXT and LABEL are objects adapted for the our data(we assume the train set is big enough to include almost all words)
    TEXT.build_vocab(train,vectors=GloVe(name='6B',dim=100),max_size=20000,min_freq=10)
    LABEL.build_vocab(train)

    #train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits((train,valid,test),batch_size=16)

    input_size = 100
    hidden_size = 512
    output_size = 5
    dropout = 0.1
    encoder = "gru"
    embedding_type = "glove"
    attention = True
    num_layers = 2

    lr=0.0001
    bsz=16
    epochs=51
    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits((train,valid,test),batch_size=bsz)
    model = classifier(input_size,hidden_size,num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    model = classifier(input_size,hidden_size,dropout=dropout,num_layers=num_layers, num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    trainer = Trainer(model, TEXT, LABEL, batch_size=bsz, lr=lr, epochs=epochs)
    trainer.train(train_iter, valid_iter, test_iter)



    input_size = 100
    hidden_size = 1024
    output_size = 5
    dropout = 0.05
    encoder = "gru"
    embedding_type = "glove"
    attention = True
    num_layers = 2

    lr=0.0001
    bsz=16
    epochs=51
    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits((train,valid,test),batch_size=bsz)
    model = classifier(input_size,hidden_size,num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    model = classifier(input_size,hidden_size,dropout=dropout,num_layers=num_layers, num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    trainer = Trainer(model, TEXT, LABEL, batch_size=bsz, lr=lr, epochs=epochs)
    trainer.train(train_iter, valid_iter, test_iter)


    input_size = 100
    hidden_size = 1024
    output_size = 5
    dropout = 0.05
    encoder = "gru"
    embedding_type = "glove"
    attention = True
    num_layers = 2

    lr=0.001
    bsz=16
    epochs=51
    train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits((train,valid,test),batch_size=bsz)
    model = classifier(input_size,hidden_size,num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    model = classifier(input_size,hidden_size,dropout=dropout,num_layers=num_layers, num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    trainer = Trainer(model, TEXT, LABEL, batch_size=bsz, lr=lr, epochs=epochs)
    trainer.train(train_iter, valid_iter, test_iter)


    # input_size = 100
    # hidden_size = 512
    # output_size = 5
    # dropout = 0.1                      
    # encoder = "transformer"
    # embedding_type = "glove"
    # attention = True
    # num_layers = 2
    

    # lr=0.0001
    # bsz=16
    # epochs=200

    # #wandb.init(project='emotion-classification',config={'input_size':input_size, 'hidden_size':hidden_size, 'encoder':encoder, 'embedding':embedding_type, 'attention':attention, 'batch_size':bsz, 'lr':lr})
    # model = classifier(input_size,hidden_size,dropout=dropout,num_layers=num_layers, num_class=output_size,encoder=encoder,embedding_type=embedding_type,attention=attention,TEXT=TEXT)
    # trainer = Trainer(model, TEXT, LABEL, batch_size=bsz, lr=lr, epochs=epochs)
    # trainer.train(train_iter, valid_iter, test_iter)

    # # wandb.finish()

