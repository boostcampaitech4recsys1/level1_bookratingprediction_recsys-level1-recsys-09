# 라이브러리 호출
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

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

    '''
    ### book category 설정 
    book['category'] = book['category'].str.lower()
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
     'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
     'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
     'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india','history','mind','fitness','art','nature',
     'self-help','reference','sports','comics','education','crime','pets','mystery','medical','technology','game','law','biography ']

    for category in categories:
        book.loc[book[book['category'].str.contains(category,na=False)].index,'category_high'] = category

    category_high_df = pd.DataFrame(book['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']


    # 5개 이하인 항목은 others로 묶어놓기
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    book.loc[book[book['category_high'].isin(others_list)].index, 'category_high']='others'

    # 카테고리는 있으나 대카테고리에 포합된 것이 없는 경우, 그대로 카테고리 값을 대카테고리 값으로 가져가기, 이때 마찬가지로 카운트가 5미만이면 other로 치환
    left_cate = book[book['category'].notnull() & book['category_high'].isnull()]['category'].value_counts().reset_index()
    l_ = left_cate[left_cate['category']<5]['index'].values
    book.loc[book[book['category_high'].isin(l_)].index, 'category_high']='others'

    for i in book[book['category'].notnull() & book['category_high'].isnull()]['category_high'].index:
        book.loc[i,'category_high'] = book.loc[i,'category'][2:-2]

    l_2 = book.category_high.value_counts().reset_index()

    book.loc[book[book['category_high'].isin(l_2[l_2['category_high']<5]['index'])].index, 'category_high']= 'others'
    # 결측치는 결측치대로 두기
    # 그외 피쳐는 의미없다고 생각해서 제외
    '''
    
    ### 지역 설정
    # 지역의 경우 세세한 지역보다는 간단한 국가 정도만 사용
    user['country'] = user['location'].apply(lambda x:x.split(',')[-1])


    ### 각 train 및 test별로 라벨 인덱싱이 필요한 경우 ~ data leakage 고려되야하는 부분 ~ 인덱싱 및 나이 통계치 확인                            
    ratings = pd.concat([train, test]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(user, on='user_id', how='left').merge(book[['isbn','year_of_publication','category']], on='isbn', how='left')
    train_df = train.merge(user, on='user_id', how='left').merge(book[['isbn','year_of_publication','category']], on='isbn', how='left')
    test_df = test.merge(user, on='user_id', how='left').merge(book[['isbn','year_of_publication','category']], on='isbn', how='left')
    
    # 나라 인덱싱
    country_idexing = {v:k for k,v in enumerate(context_df['country'].unique())}
    train_df['country'] = train_df['country'].map(country_idexing)
    test_df['country'] = test_df['country'].map(country_idexing)

    train_df = train_df.drop('location',axis=1)
    test_df = test_df.drop('location',axis=1)

    # 나이 맵핑 및 평균 사용 
    avg = int(train_df['age'].mean())
    # leakage 배제를 위한 train mean값 그대로 쓰기
    train_df['age'] = train_df['age'].fillna(avg)
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(avg)
    test_df['age'] = test_df['age'].apply(age_map)

    # 저자 인덱싱
    '''
    author_indexing = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    train_df['book_author'] = train_df['book_author'].map(author_indexing)
    test_df['book_author'] = test_df['book_author'].map(author_indexing)
    '''
    # 카테고리 인덱싱
    category_indexing = {v:k for k,v in enumerate(context_df['category'].unique())}
    train_df['category'] = train_df['category'].map(category_indexing)
    test_df['category'] = test_df['category'].map(category_indexing)

    # 발행년도 인덱싱
    year_indexing = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}
    train_df['year_of_publication'] = train_df['year_of_publication'].map(year_indexing)
    test_df['year_of_publication'] = test_df['year_of_publication'].map(year_indexing)

    # 필드 차원 수 정해주기
    field_dim = np.array([len(user2idx), len(isbn2idx), 6, len(country_idexing),len(year_indexing) ,len(category_indexing)], dtype=np.uint32)

    # 나중에 인덱싱한거 다시 되돌리기 용 및 기타 데이터 다 저장해서 넘기기 ~ data['train'] 이런식으로 조회 및 타 데이터 추가 가능하게
    data = {
            'train' : train_df,
            'test' : test_df.drop(['rating'], axis=1),
            'user':user,
            'book':book,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,  
            'field_dim' : field_dim   
            }

    return data 

def context_data_split(data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

    return data

def stratified_kfold(data,n):
    skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=seed)
    counts = 0
    for train_index, valid_index in skf.split(data['train'].drop(['rating'], axis=1),data['train']['rating']):
        if counts == n:
            data['X_train'], data['y_train'] = data['train'].drop(['rating'], axis=1).loc[train_index], data['train']['rating'].loc[train_index]
            data['X_valid'], data['y_valid'] = data['train'].drop(['rating'], axis=1).loc[valid_index], data['train']['rating'].loc[valid_index]
            break
        else:
            counts += 1
        
    return data

def context_data_loader(data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data