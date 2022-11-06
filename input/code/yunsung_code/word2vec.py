import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from word2vecModel import Word2Vec

# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/data.csv", filename="data.csv")

def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def get_document_vectors(document_list):
    document_embedding_list = []
    # 각 문서에 대해서
    index = 0
    index_array=[]
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word in word2vec_model.wv.key_to_index:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model.wv[word]
                else:
                    doc2vec = doc2vec + word2vec_model.wv[word]
        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
        else:
            index_array.append(index)
        index+=1
    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return (document_embedding_list,index_array)

def recommendations(title):
    books = df[['book_title', 'img_url']]
    rating = 0
    # 책의 제목을 입력하면 해당 제목의 인덱스를 리턴받아 idx에 저장.
    indices = pd.Series(df.index, index = df['book_title']).drop_duplicates()    
    idx = indices[title]
    if len(idx)>1:
        idx = indices[title][0]
    # 입력된 책과 줄거리(document embedding)가 유사한 책 5개 선정.
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:10]
    for i in sim_scores:
        rating+=df_total.iloc[df.iloc[i[0]]['index']]['rating']
    rating /=10
    # 가장 유사한 책 5권의 인덱스
    book_indices = [i[0] for i in sim_scores]
    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 5개의 행을 가진다.
    recommend = books.iloc[book_indices].reset_index(drop=True)

    fig = plt.figure(figsize=(20, 30))

    # 데이터프레임으로부터 순차적으로 이미지를 출력
    for index, row in recommend.iterrows():
        response = requests.get(row['img_url'])
        img = Image.open(BytesIO(response.content))
        fig.add_subplot(1, 10, index + 1)
        plt.imshow(img)
        plt.title(row['book_title'])
    return rating
df = pd.read_csv("/opt/ml/input/code/data/books.csv")
df_user = pd.read_csv("/opt/ml/input/code/data/users.csv")
df_train =pd.read_csv("/opt/ml/input/code/data/train_ratings.csv")
df_total =df_user.merge(df_train,on="user_id")
df_total =df_total.merge(df,on="isbn")
idx = df[df['summary'].isna()].index
df = df.drop(idx)
df = df.reset_index(drop=False)
df['cleaned'] = df['summary'].apply(_removeNonAscii)
df['cleaned'] = df.cleaned.apply(make_lower_case)
df['cleaned'] = df.cleaned.apply(remove_stop_words)
df['cleaned'] = df.cleaned.apply(remove_punctuation)
df['cleaned'] = df.cleaned.apply(remove_html)
df['cleaned'].replace('', np.nan, inplace=True)
df = df[df['cleaned'].notna()]
corpus = []
for words in df['cleaned']:
    corpus.append(words.split())

word2vec_model = Word2Vec(vector_size = 300, window=5, min_count = 2, workers = -1)
word2vec_model.build_vocab(corpus)
word2vec_model.wv.vectors_lockf = np.ones(len(word2vec_model.wv),dtype=np.float32)
word2vec_model.wv.intersect_word2vec_format('GoogleNews-vectors-negative300.bin.gz', lockf=1.0, binary=True)
word2vec_model.train(corpus, total_examples = word2vec_model.corpus_count, epochs = 15)
print("training complete")
document_embedding_list,delete_array=get_document_vectors(df['cleaned'])
df = df.drop(delete_array)
df = df.reset_index(drop=True)
cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)
rating = recommendations("The Da Vinci Code")
print(rating)