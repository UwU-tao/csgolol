import torch
import argparse
from torch.utils.data import DataLoader
from src.utils import *
from src import train
import numpy as np
import os

parser = argparse.ArgumentParser()
# Fixed
parser.add_argument('--model', type=str, default='AverageBERT', help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--data_path', type=str, default='data/dataset', help='path for storing the dataset')

# Dropouts
parser.add_argument('--mlp_dropout', type=float, default=0.2, help='fully connected layers dropout')

# Architecture
parser.add_argument('--bert_model', type=str, default="bert-base-cased", help='pretrained bert model to use')
parser.add_argument('--cnn_model', type=str, default="vgg16", help='pretrained CNN to use for image feature extraction')
parser.add_argument('--image_feature_size', type=int, default=1000, help='image feature size extracted from pretrained CNN (default: 4096)')
parser.add_argument('--bert_hidden_size', type=int, default=768, help='bert hidden size for each word token (default: 768)')

# Tuning
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size (default: 8)')
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

torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ratings = pd.read_csv(f"{args.data_path}/ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine='python')
movie_ids = ratings.movie_id.unique()

# grouped = ratings.groupby('movie_id')[['user_id', 'rating']].apply(lambda x: dict(zip(x['user_id'], x['rating']))).to_dict()
# res = {movie_id: [grouped.get(movie_id, {}).get(user_id, 0) for user_id in range(1, len(ratings.user_id.unique()) + 1)] for movie_id in movie_ids}
res = {}
n = len(ratings.user_id.unique())

for movie_id in movie_ids:
    tmp = np.zeros(n)
    cur_users = ratings.loc[ratings['movie_id']==movie_id].user_id.tolist()
    temp = ratings['rating']
    for user in cur_users:
        tmp[user - 1] = temp[user-1]
    res[movie_id] = tmp
        


print("Start loading the data....")
train_data = get_data(args, res, 'train')
valid_data = get_data(args, res, 'dev')
test_data = get_data(args, res, 'test')

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