
import time
import pandas as pd
import os
import random
import numpy as np
import torch 
import torch.nn as nn
import wandb


from ffm_data import context_data_loader, data_processing, stratified_kfold
from ffm_model import FieldAwareFactorizationMachineModel

book = pd.read_csv('~/input/code/data/books_pub_year.csv')
user = pd.read_csv('~/input/code/data/users_fillage.csv')
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
    config={"epochs":15,"batch_size":32,"learning_rate":0.01}
    seed_everything(9)
    data = data_processing()
    for i in range(5):
        wandb.init(project="competition1",config=config)
        wandb.run.name=f'FFM'
        data = stratified_kfold(data,i)
        data = context_data_loader(data)
        model = FieldAwareFactorizationMachineModel(data)

        model.train()
        predicts = model.predict(data['test_dataloader'])
        predicts_list.append(predicts)
        
    sub['rating'] = np.mean(predicts_list, axis=0)

    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    sub.to_csv('{}_{}.csv'.format(save_time,"FFM"), index=False)

main()