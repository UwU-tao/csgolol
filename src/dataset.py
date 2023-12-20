import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

Image.MAX_IMAGE_PIXELS = 1000000000

class MyDataset(Dataset):

    def __init__(self, root_dir, split, transform=None):
        title = []
        image = []
        genres = []

        with open(f"{root_dir}/{split}.dat") as f:
            lines = f.readlines()
            for line in lines:
                movie_id, title_, genre_, img_path = line.split(",")
                title.append(title_)
                image.append(img_path)
                genres.append(genre_.split("|"))
            
        self.data_dict = pd.DataFrame({'image': image, 'label': genres, 'text': title})
            
        self.root_dir = root_dir
        self.transform = transform
        self.genres = ["Crime", "Thriller", "Fantasy",
                       "Horror", "Sci-Fi", "Comedy",
                       "Documentary", "Adventure", "Film-Noir",
                       "Animation", "Romance", "Drama",
                       "Western", "Action", "Mystery",
                       "Musical", "War", "Children's"]
                       
        self.num_classes = len(self.genres)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_dict.iloc[idx,0]
        image = Image.open(img_name.strip()).convert('RGB')
        
        label = self.data_dict.iloc[idx,1]
        indeces = torch.LongTensor([self.genres.index(e) for e in label])
        label = torch.nn.functional.one_hot(indeces, num_classes = self.num_classes).sum(dim=0)

        text = self.data_dict.iloc[idx,2]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text,
                  "label": label.type(torch.FloatTensor)}

        return sample