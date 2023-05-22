import datetime
from re import split

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
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
    # movies = splitMultipleData(movies, "genres", "|")
    # movies['genres'] = movies['genres'].str.split('|', n=1, expand=True)
    movies['genres'] = movies['genres'].str.split('|')
    movies = movies.drop(['Unnamed: 3'], axis=1)
    movies.duplicated('title').sum()  # 18

    count = 0
    for i in movies['genres']:
        if i[0] and not genres_list.__contains__(i[0]):
            movies.loc[count, 'title'] += str(i[0])
            movies.loc[count, 'genres'].remove(i[0])
        count += 1

    # movies['MovieYear'] = movies['MovieYear'].map(lambda x: x.rstrip(')').lstrip(''))
    # movies = movies.drop(['title'],axis=1)
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
    # ratings.drop([timestamp])
    ratings['timestamp'] = ratings['timestamp'].apply(convert_timestamp)

    # split day column from date
    ratings['Day'] = pd.to_datetime(ratings['timestamp']).dt.day
    ratings['Month'] = pd.to_datetime(ratings['timestamp']).dt.month
    ratings['Year'] = pd.to_datetime(ratings['timestamp']).dt.year
    ratings['hour'] = pd.to_datetime(ratings['timestamp']).dt.hour
    ratings['minute'] = pd.to_datetime(ratings['timestamp']).dt.minute
    ratings['second'] = pd.to_datetime(ratings['timestamp']).dt.second

    ratings = ratings.drop(['timestamp'], axis=1)
    ratings.to_csv(r'ratings_test.csv', index=False)
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


def content_based_model(movies, ratings, movie_name='Client, The (1994)'):
    ratings = pd.merge(ratings, movies, on='movieId')
    ratings_avg = ratings.groupby('title')['rating'].describe()['mean']
    ratings_count = ratings.groupby('title')['rating'].describe()['count']
    avg_ratings = pd.concat([ratings_count, ratings_avg, ], axis=1)

    data = pd.DataFrame()
    data['title'] = ratings['title']
    data['genres'] = ratings['genres']
    data = data.drop_duplicates(subset=['title'])

    avg_ratings.rename(columns={'count': 'total_reviews', 'mean': 'average_rating'}, inplace=True)
    avg_ratings.reset_index()
    avg_ratings['average_rating'].plot(bins=100, kind='hist')
    avg_ratings['total_reviews'].plot(bins=100, kind='hist', color='r')
    avg_ratings[avg_ratings['average_rating'] == 5]
    avg_ratings.sort_values('total_reviews', ascending=False).head(100)
    ratings_matrix = ratings.pivot_table(index='userId', columns='title', values='rating')
    try:
        wizard_of_oz = ratings_matrix[movie_name]
        similarity_scores = pd.DataFrame(ratings_matrix.corrwith(wizard_of_oz), columns=['Similarity'])
        similarity_scores = similarity_scores.join(avg_ratings['total_reviews'])

        similarity_scores.dropna(inplace=True)
        # sort by highest to lowest similarity
        similarity_scores.sort_values('Similarity', ascending=False)

        # get the top 10 similar movies
        top_movies = similarity_scores[similarity_scores['total_reviews'] >= 80].sort_values('Similarity', ascending=False).head(10)


        # create a bar chart of the top 10 similar movies
        fig, ax = plt.subplots()
        ax.barh(top_movies.index, top_movies['Similarity'], align='center')

        # set the y-axis label and title
        ax.set_ylabel("Movie Title")
        ax.set_title("Top 10 Similar Movies to: " + movie_name)

        # #show the plot
        plt.show()


        return top_movies.iloc[1:11]
    except Exception as e:
        print(e)
        return "No movies found. Please check your input"






def item_based_model(movies, movie_name):
    movies.drop(['genres'], axis=1)
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
    final_dataset.fillna(0, inplace=True)

    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

    f, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_user_voted.index, no_user_voted, color='mediumseagreen')
    plt.axhline(y=10, color='r')
    plt.xlabel('MovieId')
    plt.ylabel('No. of users voted')
    # plt.show()

    final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 40].index, :]
    f, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_movies_voted.index, no_movies_voted, color='mediumseagreen')
    plt.axhline(y=50, color='r')
    plt.xlabel('UserId')
    plt.ylabel('No. of votes by user')
    # plt.show()
    final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 90].index]

    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    return get_movie_recommendation(final_dataset, knn, csr_data, movie_name)


def get_movie_recommendation(final_dataset, knn, csr_data, movie_name='Toy Story'):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append(
                {'movieId': movies.iloc[idx]['movieId'].values[0], 'Title': movies.iloc[idx]['title'].values[0],
                 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
        df = df.sort_values('Distance').reset_index()
        df.index = df.index + 1
        print("The Recommendation based on Movie :", movie_list.iloc[0]['title'])
        df = df.drop(['index'], axis=1)
        return df
    else:
        return "No movies found. Please check your input"


def predict_user_movie_rating(user_id, movie_id):
    movies.drop(['genres'], axis=1)
    final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
    final_dataset.fillna(0, inplace=True)
    final_dataset.reset_index(inplace=True)

    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
    knn.fit(final_dataset.values)

    distances, indices = knn.kneighbors(final_dataset.values, n_neighbors=5)

    index_for_movie = final_dataset.index.tolist().index(movie_id)
    sim_movies = indices[index_for_movie].tolist()
    movie_distances = distances[index_for_movie].tolist()
    id_movie = sim_movies.index(index_for_movie)
    sim_movies.remove(index_for_movie)
    movie_distances.pop(id_movie)

    print('The Nearest Movies to movie_0:', sim_movies)
    print('The Distance from movie_0:', movie_distances)

    movie_similarity = [-x + 1 for x in movie_distances]  # inverse distance

    predicted_rating = sum(
        movie_similarity[i] * final_dataset.iloc[sim_movies[i], user_id] for i in range(len(sim_movies))
    ) / sum(movie_similarity)

    print(predicted_rating)

    return


# ************************************************************* #


movies = movies_pre_processing()
ratings = ratings_pre_processing()
users = users_pre_processing()

user_id = 2

# model = int(input("Choose model: \n1.content-based \n2.item-based\n "))
# movie = str(input("Enter movie name\n"))
model = 1
movie = "Client, The (1994)"
if model == 1:
    ret = content_based_model(movies, ratings, movie)
    #
    # newDf = pd.DataFrame(ret)
    # plt.bar(newDf['total_reviews'], newDf['Similarity'], color='green')
    # plt.xlabel('Movie Title')
    # plt.ylabel('Total Reviews')
    # plt.title('Top-rated Movies based on Total Reviews')
    # plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    # plt.show()
    print(ret)
else:
    ret = item_based_model(movies, movie)
    print(ret)

predict_user_movie_rating(1193, 1)
