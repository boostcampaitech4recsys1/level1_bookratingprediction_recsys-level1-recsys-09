from catboost import CatBoostRegressor
from sklearn.metrics import *
from LGBM_data_processing import *
import os
import time
import warnings
warnings.filterwarnings(action='ignore')


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

def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))
    
def main():
    seed_everything(9)
    data = data_processing(train,test,sub,book,user)
    
    predicts_list = []
    cat_features = ['category', 'publisher', 'language', 'book_author','age','location_city','location_state','location_country']
    for i in  range(5):
        model = CatBoostRegressor(iterations=1000, random_state=9, eval_metric='RMSE') 
        data = stratified_kfold(data,i)
        evals = [(data['X_valid'],data['y_valid'])]
        model.fit(data['X_train'], data['y_train'], eval_set= evals, early_stopping_rounds=300, cat_features=cat_features, verbose=100)
  
        lgbm_preds = model.predict(data['X_valid'])
        print(rmse(data['y_valid'].tolist(),lgbm_preds.tolist()))
        predicts_list.append(model.predict(data['test']))

    
    sub['rating'] = np.mean(predicts_list, axis=0)
    
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    sub.to_csv('{}_{}.csv'.format(save_time,"lgbm"), index=False)

    
main()