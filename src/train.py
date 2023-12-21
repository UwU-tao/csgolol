from src import models
from src.utils import *
import numpy as np
import time
import sys

from transformers import BertModel
from transformers import BertTokenizer

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torchmetrics.classification import MultilabelF1Score, MultilabelRecall, MultilabelPrecision
from tqdm import tqdm

from src.metrics import *

import os


def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)
    bert = BertModel.from_pretrained(hyp_params.bert_model)
    tokenizer = BertTokenizer.from_pretrained(hyp_params.bert_model)

    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', hyp_params.cnn_model, pretrained=True)
    for param in feature_extractor.features.parameters():
        param.requires_grad = False

    model.to(hyp_params.device)
    bert.to(hyp_params.device)
    feature_extractor.to(hyp_params.device)

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    linear = nn.Linear(786, 20)
    
    settings = {'model': model,
                'bert': bert,
                'tokenizer': tokenizer,
                'feature_extractor': feature_extractor,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler,
                'linear': linear}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    bert = settings['bert']
    tokenizer = settings['tokenizer']
    feature_extractor = settings['feature_extractor']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    scheduler = settings['scheduler']

    linear = settings['linear']
    
    def train(model, bert, tokenizer, feature_extractor, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        total_loss = 0.0
        losses = []
        results = []
        truths = []

        for data_batch in tqdm(train_loader):
            
            input_ids = data_batch["input_ids"]
            targets = data_batch["label"]
            images = data_batch['image']
            
            text_encoded = tokenizer(input_ids, padding=True, return_tensors='pt').to(hyp_params.device)

            input_ids = text_encoded['input_ids'].clone().detach().to(hyp_params.device)
            attention_mask = text_encoded['attention_mask'].clone().detach().to(hyp_params.device)
            
            targets = targets.to(hyp_params.device)
            images = images.to(hyp_params.device)
            
            if images.size()[0] != input_ids.size()[0]:
                continue
            
            with torch.no_grad():
                feature_images = feature_extractor.features(images)
                feature_images = feature_extractor.avgpool(feature_images)
                feature_images = torch.flatten(feature_images, 1)
                feature_images = feature_extractor.classifier[0](feature_images)

            with torch.no_grad():
              outs = bert(**text_encoded)
            
            optimizer.zero_grad()
            outputs = model(
                last_hidden=outs.last_hidden_state,
                pooled_output=outs.pooler_output,
                feature_images=feature_images
            )
    
            preds = outputs
            preds_round = (preds > 0.5).float()
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            total_loss += loss.item() * hyp_params.batch_size
            results.append(preds)
            truths.append(targets)
                
        avg_loss = total_loss / hyp_params.n_train
        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    def evaluate(model, bert, tokenizer, feature_extractor, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        correct_predictions = 0

        with torch.no_grad():
            for data_batch in loader:
                
                input_ids = data_batch["input_ids"]
                targets = data_batch["label"]
                images = data_batch['image']
                
                text_encoded = tokenizer(input_ids, padding=True, return_tensors='pt').to(hyp_params.device)
 
                input_ids = text_encoded['input_ids'].clone().detach().to(hyp_params.device)
                attention_mask = text_encoded['attention_mask'].clone().detach().to(hyp_params.device)

                targets = targets.to(hyp_params.device)
                images = images.to(hyp_params.device)

                if images.size()[0] != input_ids.size()[0]:
                    continue

                with torch.no_grad():
                    feature_images = feature_extractor.features(images)
                    feature_images = feature_extractor.avgpool(feature_images)
                    feature_images = torch.flatten(feature_images, 1)
                    feature_images = feature_extractor.classifier[0](feature_images)
                
                with torch.no_grad():
                  outs = bert(**text_encoded)
                
                outputs = model(
                    last_hidden=outs.last_hidden_state,
                    pooled_output=outs.pooler_output,
                    feature_images=feature_images
                )
                preds = outputs
                
                total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
                correct_predictions += torch.sum(preds == targets)

                # Collect the results into dictionary
                results.append(preds)
                truths.append(targets)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    best_valid = 1e8
    # writer = SummaryWriter('runs/'+hyp_params.model)
    
    for epoch in range(1, hyp_params.num_epochs+1):
        train_results, train_truths, train_loss = train(model, bert, tokenizer, feature_extractor, optimizer, criterion)
        val_results, val_truths, val_loss = evaluate(model, bert, tokenizer, feature_extractor, criterion, test=False)
        
        scheduler.step(val_loss)

        train_acc, train_prec, train_recall, train_f1 = metrics(train_results, train_truths)
        val_acc, val_prec, val_recall, val_f1 = metrics(val_results, val_truths)
        
        # print("-"*156)
        # print('Epoch {:2d} | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f} | Valid f1-score {:5.4f}'.format(epoch, train_loss, val_loss, val_acc, val_f1))
        # print("-"*156)
        if epoch == 1:
            print(f'Epoch  |     Train Loss     |     Train Accuracy     |     Valid Loss     |     Valid Accuracy     |     Precision     |     Recall     |     F1-Score     |')
        
        print(f'{epoch:^7d}|{train_loss:^20.4f}|{train_acc:^24.4f}|{val_loss:^20.4f}|{val_acc:^24.4f}|{val_prec:^19.4f}|{val_recall:^16.4f}|{val_f1:^18.4f}|')

        if val_loss < best_valid:
            print(f"Saved model at pretrained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if test_loader is not None:
        model = load_model(hyp_params, name=hyp_params.name)
        results, truths, val_loss = evaluate(model, bert, tokenizer, feature_extractor, criterion, test=True)
        test_acc, test_prec, test_recall, test_f1 = metrics(results, truths)
        
        print("\n\nTest Acc {:5.4f} | Test f1-score {:5.4f}".format(test_acc, test_f1))

    sys.stdout.flush()
    