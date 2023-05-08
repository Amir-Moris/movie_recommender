import datetime
from re import split

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

genres_list = [
        'Action',
        'Adventure',
        'Animation',
        "Children's",
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Fantasy',
        'Film-Noir',
        'Horror',
        'Musical',
        'Mystery',
        'Romance',
        'Sci-Fi',
        'Thriller',
        'War',
        'Western', ]
def splitMultipleData(df, column, spliter):


    # create a new DataFrame with columns for each language
    df_encoded = pd.DataFrame(columns=genres_list)

    # encode each row
    for i, row in df.iterrows():
        genres = row[column]
        if pd.isna(genres):
            encoding = [0] * len(genres_list)
        else:
            genres = genres.split(spliter)
            encoding = [1 if x in genres else 0 for x in genres_list]
        df_encoded.loc[i] = encoding

    # merge the original dataframe with the encoded dataframe
    df_final = pd.concat([df, df_encoded], axis=1)

    # drop the original 'Languages' column
    df_final.drop(columns=[column], inplace=True)

    return df_final

def movies_pre_processing():
    # no null values or duplicated rows
    movies = pd.read_csv('movies.csv', sep=';', encoding='latin-1')
    movies = splitMultipleData(movies, "genres", "|")
    movies = movies.drop(['Unnamed: 3'], axis=1)
    movies[['MovieName', 'MovieYear']] = movies['title'].str.rsplit('(', n=1, expand=True)


    count = 0
    for i in movies['MovieYear']:
        if i and not i.endswith(')'):
            movies.loc[count,'MovieName'] += str(i)
            # movies.loc[count,'MovieName'] = str(movies.loc[count-1,'MovieName'])
            movies.loc[count, 'MovieYear'] = str(movies.loc[count - 1, 'MovieYear'])
        elif not i:
            movies.loc[count,'MovieYear'] = str(movies.loc[count-1,'MovieYear'])

        count += 1

    movies['MovieYear'] = movies['MovieYear'].map(lambda x: x.rstrip(')').lstrip(''))
    movies = movies.drop(['title'],axis=1)
    movies.to_csv(r'moives_test.csv', index=False)
    return movies


def users_pre_processing():
    # no null values or duplicated rows
    users = pd.read_csv('users.csv', sep=';', encoding='latin-1')
    # users.to_csv(r'users_test.csv', index=False)
    return users

def ratings_pre_processing():
    # no null values or duplicated rows
    ratings = pd.read_csv('ratings.csv', sep=';', encoding='latin-1')
    ratings.to_csv(r'ratings_test.csv', index=False)
    print(ratings.isna().sum())
    print(ratings.duplicated().sum())

    ratings['timestamp'] = ratings['timestamp'].apply(convert_timestamp)

    # split day column from date
    ratings['Day'] = pd.to_datetime(ratings['timestamp']).dt.day
    ratings['Month'] = pd.to_datetime(ratings['timestamp']).dt.month
    ratings['Year'] = pd.to_datetime(ratings['timestamp']).dt.year
    ratings['hour'] = pd.to_datetime(ratings['timestamp']).dt.hour
    ratings['minute'] = pd.to_datetime(ratings['timestamp']).dt.minute
    ratings['second'] = pd.to_datetime(ratings['timestamp']).dt.second
    ratings = ratings.drop(['timestamp'], axis=1)
    # ratings.to_csv(r'ratings_test.csv', index=False)
    return ratings


def convert_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def split_movie_name(str):
    return split(r'(\d\d\d\d)', str)

def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X

def magic(df):
    list = []
    for gen in genres_list:
        list.append(df.loc[gen])
    return list


# ************************************************************* #

movies = movies_pre_processing()
ratings = ratings_pre_processing()
users = users_pre_processing()


# Split the genres into a list of individual genres
# movies['genres'] = movies['genres'].str.split('|')

# Create a MultiLabelBinarizer object
# mlb = MultiLabelBinarizer()

# Fit and transform the genres data with the MultiLabelBinarizer
# genre_features = mlb.fit_transform(movies['genres'])

# Convert the genre features matrix to a dataframe
# genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)

# Concatenate the genre features dataframe with the original data
# movies = pd.concat([movies, genre_df], axis=1)

# Merge the movie data with the ratings data
data = pd.merge(ratings, movies, on='movieId')

# Merge the user data with the ratings data
data = pd.merge(data, users, on='userId')

# Convert the movie genre and year to string and concatenate them to create the content feature
# + ' ' + movies['MovieYear'].astype(str)
# data['binary_genres'] = str(data[genres_list].apply(lambda x: ''.join(str(x)), axis=1))
# data['binary_genres']=[]

# data['binary_genres'] = genres_list
# data['binary_genres'] = data['binary_genres'].str.cat(magic(data), sep=' ')
# data['binary_genres'] = pd.concat([data[col] for col in genres_list], axis=1)
data[genres_list] = data[genres_list].astype(str)
data['new_col'] = data[genres_list].apply(lambda x: ''.join(x), axis=1)
# movies['content'] = content_features
# data = magic(data)
data.to_csv(r'data.csv', index=False)
movies.to_csv(r'moives_test.csv', index=False)

# genre_df.to_csv(r'concat.csv',index=False)


# # Split the genres into a list of individual genres
# movies['genres'] = movies['genres'].str.split('|')
#
# # Convert the list of genres to a string with spaces between each genre
# movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
#
# # Merge the movie data with the ratings data
# data = pd.merge(ratings, movies, on='movieId')
#
# # Convert the movie genre and year to string and concatenate them to create the content feature
# data['content'] = data['genres'].astype(str) + ' ' + data['MovieYear'].astype(str)
#
# # Merge the user data with the ratings data
# data = pd.merge(data, users, on='userId')


#
# movies['genres'] = movies['genres'].str.split('|')
#
# mlb = MultiLabelBinarizer()
#
# # Fit and transform the genres data with the MultiLabelBinarizer
# genre_features = mlb.fit_transform(movies['genres'])
#
# # Convert the genre features matrix to a dataframe
# genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)
#
# # Concatenate the genre features dataframe with the original data
# movies = pd.concat([movies, genre_df], axis=1)
#
#
# # Merge the movie data with the ratings data
# data = pd.merge(ratings, movies, on='movieId')
#


# Convert the movie genre and year to string and concatenate them to create the content feature
# data['content'] = data['genres'].astype(str) + ' ' + data['MovieYear'].astype(str)
# data['content'] = movies['genres'].astype(str) + ' ' + movies['MovieYear'].astype(str)


# Merge the user data with the ratings data
# data = pd.merge(data, users, on='userId')


