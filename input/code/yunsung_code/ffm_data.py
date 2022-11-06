import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6


def process_context_data(users, books, ratings1, ratings2):
    # users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    # users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    # users['location_country'] = users['location_country'].apply(lambda x: x.split(',')[0])
    # users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'book_author','publisher','year_of_publication']], on='isbn', how='left') #, 'book_author','language'
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'book_author','publisher','year_of_publication']], on='isbn', how='left') # , 'language', 'book_author'
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'book_author','publisher','year_of_publication']], on='isbn', how='left') #  'language', 'book_author'

    # 인덱싱 처리
    # loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    # loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    # train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    # train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    # test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    # test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    age2idx = {v:k for k,v in enumerate(context_df['age'].unique())}
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    # language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    year2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    # train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df['year_of_publication'] = train_df['year_of_publication'].map(year2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    # test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df['year_of_publication'] = test_df['year_of_publication'].map(year2idx)

    idx = {
        # "loc_city2idx":loc_city2idx,
        # "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "age2idx":age2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        # "language2idx":language2idx,
        "author2idx":author2idx,
        "year2idx":year2idx,
    }

    return idx, train_df, test_df

def data_processing():

    ######################## DATA LOAD
    books = pd.read_csv('~/input/code/data/books_pub_year.csv')
    users = pd.read_csv('~/input/code/data/users_fillage.csv')
    train = pd.read_csv('~/input/code/data/train_ratings.csv')
    test = pd.read_csv('~/input/code/data/test_ratings.csv')
    sub = pd.read_csv('~/input/code/data/sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            len(idx['age2idx']), len(idx['loc_country2idx']), # len(idx['loc_city2idx']), len(idx['loc_state2idx']),
                            len(idx['category2idx']), len(idx['author2idx']), len(idx['publisher2idx']), len(idx['year2idx'])], dtype=np.uint32) # ,len(idx['language2idx']), len(idx['author2idx'])

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

def context_data_loader(data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data

def stratified_kfold(data,n):
    skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=9)
    counts = 0
    for train_index, valid_index in skf.split(data['train'].drop(['rating'], axis=1),data['train']['rating']):
        if counts == n:
            data['X_train'], data['y_train'] = data['train'].drop(['rating'], axis=1).loc[train_index], data['train']['rating'].loc[train_index]
            data['X_valid'], data['y_valid'] = data['train'].drop(['rating'], axis=1).loc[valid_index], data['train']['rating'].loc[valid_index]
            break
        else:
            counts += 1
        
    return data