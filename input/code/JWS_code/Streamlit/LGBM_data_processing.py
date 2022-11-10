# 라이브러리 호출
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.preprocessing import LabelEncoder


# 시드값 고정
seed = 9
random.seed(seed)
np.random.seed(seed)

# 나이대 맵핑
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


def data_processing(train,test,sub,book,user):
    ### 데이터 전처리 부분
    # 아이디 값 인덱싱 ~ 보기 편하고 이용하기 편하게 ~ data leakage가 고려되지 않는 책 아이디(isbn)와 유저 아이디에 대해서
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    user['user_id'] = user['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    book['isbn'] = book['isbn'].map(isbn2idx)

    ### 지역 설정
    user['location_country'] = user['location'].apply(lambda x: x.split(',')[2])
    user = user.drop(['location'], axis=1)


    ### 각 train 및 test별로 라벨 인덱싱이 필요한 경우 ~ data leakage 고려되야하는 부분 ~ 인덱싱 및 나이 통계치 확인                            
    ratings = pd.concat([train, test]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(user, on='user_id', how='left').merge(book[['isbn']], on='isbn', how='left')
    train_df = train.merge(user, on='user_id', how='left').merge(book[['isbn']], on='isbn', how='left')
    test_df = test.merge(user, on='user_id', how='left').merge(book[['isbn']], on='isbn', how='left')
       # 인덱싱 처리
    
    train_df.drop('user_id', inplace=True, axis=1)
    test_df.drop('user_id', inplace=True, axis=1)
    
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    train_df.drop(['rating'], axis=1 , inplace=True)
    
    # 나중에 인덱싱한거 다시 되돌리기 용 및 기타 데이터 다 저장해서 넘기기 ~ data['train'] 이런식으로 조회 및 타 데이터 추가 가능하게
    data = {
            'train' : train_df,
            'test' : test_df.drop(['rating'], axis=1),
            'user':user,
            'book':book,
            'sub':sub,
            'map_country':loc_country2idx,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'idx2isbn':idx2isbn
            }

    return data 

def context_data_split(data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['isbn'], axis=1),
                                                        data['train']['isbn'],
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

    return data

def stratified_kfold(data,n):
    skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=seed)
    counts = 0
    for train_index, valid_index in skf.split(data['train'].drop(['isbn'], axis=1),data['train']['isbn']):
        if counts == n:
            data['X_train'], data['y_train'] = data['train'].drop(['isbn'], axis=1).loc[train_index], data['train']['isbn'].loc[train_index]
            data['X_valid'], data['y_valid'] = data['train'].drop(['isbn'], axis=1).loc[valid_index], data['train']['isbn'].loc[valid_index]
            break
        else:
            counts += 1
        
    return data