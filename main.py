import torch
import argparse
from torch.utils.data import DataLoader
from src.utils import *
from src import train

import os

parser = argparse.ArgumentParser()
# Fixed
parser.add_argument('--model', type=str, default='AverageBERT', help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--data_path', type=str, default='data/dataset', help='path for storing the dataset')

# Dropouts
parser.add_argument('--mlp_dropout', type=float, default=0.2, help='fully connected layers dropout')

# Architecture
parser.add_argument('--bert_model', type=str, default="bert-base-uncased", help='pretrained bert model to use')
parser.add_argument('--cnn_model', type=str, default="vgg16", help='pretrained CNN to use for image feature extraction')
parser.add_argument('--image_feature_size', type=int, default=4096, help='image feature size extracted from pretrained CNN (default: 4096)')
parser.add_argument('--bert_hidden_size', type=int, default=20, help='bert hidden size for each word token (default: 768)')

# Tuning
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 8)')
parser.add_argument('--max_token_length', type=int, default=10, help='max number of tokens per sentence (default: 50)')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate (default: 2e-5)')
parser.add_argument('--optim', type=str, default='AdamW', help='optimizer to use (default: AdamW)')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs (default: 3)')
parser.add_argument('--when', type=int, default=2, help='when to decay learning rate (default: 2)')

# Logistics
parser.add_argument('--log_interval', type=int, default=100, help='frequency of result logging (default: 100)')
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--name', type=str, default='model', help='name of the trial (default: "model")')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers to use for DataLoaders (default: 2)')

args = parser.parse_args()

torch.manual_seed(args.seed)

output_dim = 18

criterion = 'CrossEntropyLoss'

torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Start loading the data....")
train_data = get_data(args, 'train')
valid_data = get_data(args, 'dev')
test_data = get_data(args, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print('Finish loading the data....')


hyp_params = args
hyp_params.device = device
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
    
hyp_params.model = args.model.strip()
hyp_params.output_dim = output_dim
hyp_params.criterion = criterion

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)