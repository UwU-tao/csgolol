import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from nltk import wordpunct_tokenize

Image.MAX_IMAGE_PIXELS = 1000000000

class Vocab:
  def __init__(self, freq_threshold, max_size):
    self.freq_threshold = freq_threshold
    self.max_size = max_size
    self.itos = {0:'<pad>', 1:'<sos>', 2:'<eos>', 3:'<unk>'}
    self.stoi = {j:i for i, j in self.itos.items()}


  def __len__(self):
    return len(self.itos)


  @staticmethod
  def tokenizer(text):
    return wordpunct_tokenize(text.lower())


  def build_vocab(self, sentence_list):
    freqs = {}
    idx = 4


    for sentence in sentence_list:
      for word in self.tokenizer(sentence):
        if word not in freqs.keys():
          freqs[word] = 1
        else:
          freqs[word] += 1


    freqs = {k:v for k,v in freqs.items() if v > self.freq_threshold}

    if len(freqs) > self.max_size-idx:
      freqs = dict(sorted(freqs.items(), key = lambda x : -x[1])[:self.max_size-idx])

    for word in freqs.keys():
      self.stoi[word] = idx
      self.itos[idx] = word
      idx += 1


  def numericalize(self, text):
    tokenized_text = self.tokenizer(text)
    numericalized_text = []
    for token in self.tokenizer(text):
      if token in self.stoi.keys():
        numericalized_text.append(self.stoi[token])
      else:
        numericalized_text.append(self.stoi['<unk>'])

    return numericalized_text


class MyDataset(Dataset):

    def __init__(self, root_dir, split, ratings, transform=None):
        title = []
        image = []
        genres = []
        ids = []
        
        with open(f"{root_dir}/{split}.dat") as f:
            lines = f.readlines()
            for line in lines:
                movie_id, title_, genre_, img_path = line.split(",")
                ids.append(movie_id)
                title.append(title_)
                image.append(img_path)
                genres.append(genre_.split("|"))
            
        self.data_dict = pd.DataFrame({'image': image, 'label': genres, 'text': title, 'id': ids})
        
        
        self.root_dir = root_dir
        self.transform = transform
        self.genres = ["Crime", "Thriller", "Fantasy",
                       "Horror", "Sci-Fi", "Comedy",
                       "Documentary", "Adventure", "Film-Noir",
                       "Animation", "Romance", "Drama",
                       "Western", "Action", "Mystery",
                       "Musical", "War", "Children's"]
                       
        self.num_classes = len(self.genres)
        self.ratings = ratings
        ### For LSTM
        self.vocab = Vocab(freq_threshold=5, max_size=1000)
        self.vocab.build_vocab(self.data_dict.text.values)
        self.max_title_size = 0
        for x in self.data_dict.text.values:
            self.max_title_size = max(self.max_title_size, len(x))
        
        
        ###        
        
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
        ### For LSTM
        title_tensor = [self.vocab.stoi['<sos>']]
        title_tensor += self.vocab.numericalize(text)
        title_tensor.append(self.vocab.stoi['<eos>'])
        while (len(title_tensor) < self.max_title_size):
            title_tensor.append(self.vocab.stoi['<pad>'])
        title_tensor = torch.Tensor(title_tensor, dtype=torch.long)
        ###
        
        if self.transform:
            image = self.transform(image)
        
        movie_id = int(self.data_dict.iloc[idx,3])
        # temp = self.ratings[self.ratings.movie_id == int(self.data_dict.iloc[idx,3])]
        # ratings = [0] * len(self.ratings.user_id.unique())
        # for x in range(len(self.ratings.user_id.unique())):
        #     if x in temp.user_id.values:
        #         ratings[x] = temp[temp.user_id == x].rating.values[0]
        
        sample = {'image': image,
                  'input_ids': title_tensor,
                  "label": label.type(torch.FloatTensor),
                  "ratings": torch.FloatTensor(self.ratings[movie_id]),}

        return sample