import streamlit as st
from lightgbm import LGBMRegressor
import joblib
import pandas as pd
from LGBM_data_processing import data_processing, age_map
from confirm_button_hack import cache_on_button_press

st.set_page_config(layout="wide")

book = pd.read_csv('~/input/code/data/books_word2vec.csv')
user = pd.read_csv('~/input/code/data/users.csv')
train = pd.read_csv('~/input/code/data/train_ratings.csv')
test = pd.read_csv('~/input/code/data/test_ratings.csv')
sub = pd.read_csv('~/input/code/data/sample_submission.csv')

df = data_processing(train,test,sub,book,user)
load_model = joblib.load('lgbm.pkl')

st.title("Book Recommendation")
st.header("by JWS")
country = st.selectbox("Select your country",(list(user.sort_values(by='location_country').location_country.unique())))
age = age_map(st.number_input("Write your age",step=1))
  
@cache_on_button_press('Make sure all Inputs')
def main():
   predictions = load_model.predict(pd.DataFrame({'age': [age],'location_country': [df['map_country'][country]]}))
   st.subheader("Book recommend for you")
   st.write(df['book'][df['book']['isbn']==predictions[0]]['book_title'].values[0],"    by    ",df['book'][df['book']['isbn']==predictions[0]]['book_author'].values[0])
   st.image(df['book'][df['book']['isbn']==predictions[0]]['img_url'].values[0],width=100)
  

main()
    

    


#streamlit run main.py --server.fileWatcherType None --server.port=30001