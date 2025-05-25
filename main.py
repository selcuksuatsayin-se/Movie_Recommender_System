import pandas as pd

movies = pd.read_csv('dataset.csv')

movies.head(10)

movies.describe()

movies.info()

movies.isnull().sum()

movies.columns

movies = movies[['id', 'title', 'overview', 'genre']]

movies

movies['tags'] = movies['overview'] + movies['genre']

movies

new_data = movies.drop(columns=['overview', 'genre'])

new_data

# 1.Bag of words 2.TFIDF

from sklearn.feature_extraction.text import CountVectorizer