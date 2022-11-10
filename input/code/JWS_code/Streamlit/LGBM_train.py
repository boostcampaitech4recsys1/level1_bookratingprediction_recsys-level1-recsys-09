import os
import time
import warnings

import joblib
from LGBM_data_processing import *
from lightgbm import LGBMClassifier
from sklearn.metrics import *

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
    
    model = LGBMClassifier(random_state=9,learning_rate=1e-1) 
    model.fit(data['train'][:10000].drop(['isbn'],axis=1), data['train'][:10000]['isbn'], verbose=False)

    joblib.dump(model, "lgbm_real_real.pkl")
main()

print()
