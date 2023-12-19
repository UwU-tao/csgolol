from src.dataset import *
from torchvision import transforms
import torch
import os


def get_data(args, split='train'):
    data = MyDataset(args.data_path, split, transform=transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]))
    return data


def save_model(args, model, name=''):
    name = name if len(name) > 0 else 'default_model'
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model