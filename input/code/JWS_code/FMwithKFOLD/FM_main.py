import time
import pandas as pd
import os
import random
import numpy as np
import torch 
import torch.nn as nn



from FM_data_processing import data_processing, context_data_split, context_data_loader, stratified_kfold
from FM_train import FactorizationMachineModel

book = pd.read_csv('~/input/code/data/books_word2vec.csv')
user = pd.read_csv('~/input/code/data/users.csv')
train = pd.read_csv('~/input/code/data/train_ratings.csv')
test = pd.read_csv('~/input/code/data/test_ratings.csv')
sub = pd.read_csv('~/input/code/data/sample_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

predicts_list = []
def main():
    seed_everything(9)
    data = data_processing(train,test,sub,book,user)
    for i in range(5):
        model = FactorizationMachineModel(data)
        data = stratified_kfold(data,i)
        data = context_data_loader(data)

        model.train(data)
        predicts = model.predict(data['test_dataloader'])
        predicts_list.append(predicts)
        
    sub['rating'] = np.mean(predicts_list, axis=0)

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    sub.to_csv('{}_{}.csv'.format(save_time,"FM"), index=False)

main()