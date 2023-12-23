from src.dataset import *
from torchvision import transforms
import torch
import os


def get_data(args, split='train'):
    ratings = pd.read_csv(f"{args.data_path}/ratings.dat", sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine='python')
    movie_ids = ratings.movie_id.unique()
    
    res = {}
    tmp = [0] * len(ratings.user_id.unique())
    for x in movie_ids:
        temp = ratings[ratings.movie_id == x]
        for y in range(len(ratings.user_id.unique())):
            if y in temp.user_id.values:
                tmp[y] = temp[temp.user_id == y].rating.values[0]
        res[x] = tmp
        tmp = [0] * len(ratings.user_id.unique())
    
    data = MyDataset(args.data_path, split, res, transform=transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]))
    return data


def save_model(args, model, name=''):
    name = name if len(name) > 0 else 'default_model'
    torch.save(model, f'pretrained_models/{name}.pt')


def load_model(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    model = torch.load(f'pretrained_models/{name}.pt')
    return model