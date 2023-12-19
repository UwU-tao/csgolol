import pandas as pd
import numpy as np
import os

users = pd.read_csv('data/dataset/users.dat', sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')

ratings = pd.read_csv('data/dataset/ratings.dat', engine='python',
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])

movies_train = pd.read_csv('data/dataset/movies_train.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')

movies_test = pd.read_csv('data/dataset/movies_test.dat', engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')

movies_train['genre'] = movies_train.genre.str.split('|')
movies_test['genre'] = movies_test.genre.str.split('|')

users.age = users.age.astype('category')
users.gender = users.gender.astype('category')
users.occupation = users.occupation.astype('category')
ratings.movieid = ratings.movieid.astype('category')
ratings.userid = ratings.userid.astype('category')




folder_img_path = 'data'

movies_train.reset_index(inplace=True)
movies_train['img_path'] = movies_train.apply(lambda row: os.path.join(folder_img_path, f'{row.movieid}.jpg'), axis = 1)

movies_test.reset_index(inplace=True)
movies_test['img_path'] = movies_test.apply(lambda row: os.path.join(folder_img_path, f'{row.movieid}.jpg'), axis = 1)

movies_train, movies_dev = movies_train[:int(0.8*len(movies_train))].reset_index(drop=True), movies_train[int(0.8*len(movies_train)):].reset_index(drop=True)


def title_normalizer(text):
    text = text[:-5].strip()
    text = text[:text.find('(')].strip()
    if len(text.split(',')) > 1:
        text = text.split(',')[1].strip() + ' ' + text.split(',')[0].strip()
    return text

movies_train.loc[:, 'title'] = movies_train['title'].apply(lambda x: title_normalizer(x))
movies_dev.loc[:, 'title'] = movies_dev['title'].apply(lambda x: title_normalizer(x))
movies_test.loc[:, 'title'] = movies_test['title'].apply(lambda x: title_normalizer(x))

data = {
    'train': movies_train,
    'dev': movies_dev,
    'test': movies_test
}

for ftype in data.keys():
  data[ftype].reset_index(drop=True, inplace=True)
  idx = []
  for i in range(len(data[ftype])):
      if not os.path.isfile('/content/ml1m/content/dataset/ml1m-images/' + str(data[ftype].iloc[i]['movieid']) + '.jpg'):
          idx.append(i)

  data[ftype].drop(idx, inplace=True)
  data[ftype].reset_index(drop=True, inplace=True)
  
data['train'].to_csv('ml1m/content/dataset/train.dat', index=False)
data['dev'].to_csv('ml1m/content/dataset/dev.dat', index=False)