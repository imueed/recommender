""" This file consist the impelemtation of item based collaborative filtering. """
#%%
import numpy as np
import pandas as pd

root = '/home/tuomas/Python/DATA.ML.360/ml-latest-small/'

df_movies = pd.read_csv(root+'movies.csv', usecols=['movieId', 'title'],
                        dtype={'movieId':'int32', 'title':'str'})

df_ratings = pd.read_csv(root+'ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId':'int32', 'movieId':'int32', 'rating':'float32'})

#%%
from scipy.sparse import csr_matrix
# Pivot ratings into movie features
df_movie_features = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Create sparse feature matrix out of dataframe
mat_movie_features = csr_matrix(df_movie_features.values)

#%%
# Get all seen and unseen movies for user
user_id = 2
user_unrated_movies_idx = np.where(mat_movie_features.getrow(user_id).toarray().ravel()==0)[0]
user_rated_movies_idx = np.where(mat_movie_features.getrow(user_id).toarray().ravel()!=0)[0]
user_rated_movies_r = mat_movie_features.getrow(user_id).toarray().ravel()[user_rated_movies_idx]

#%%
# Item based collaborative filtering
from numpy.linalg import norm

# Cosine difference (angle) between two vectors sim(a, b)
# Larger is better
# -1 : vectors are opposite
# 0 : vectors are orthogonal
# 1 : vectors are same
def cosine_diff(a, b):
    return (a@b) / (norm(a)*norm(b))

def get_sim(p, rated_item_u):
    p_vec = mat_movie_features.getcol(p).toarray().ravel()
    
    similarity_values = []
    for i in rated_item_u:
        i_vec = mat_movie_features.getcol(i).toarray().ravel()
        sim_ip = cosine_diff(i_vec, p_vec)
        similarity_values.append(sim_ip)
        
    return np.array(similarity_values)

def calc_pred(sim_vec, r_vec):
    num = np.inner(sim_vec, r_vec)
    denom = np.sum(sim_vec)
    return num / denom

# Loop over unrated movies
unseen_movies_predictions = []
n=1
for p in user_unrated_movies_idx:
    similarity_values = get_sim(p, user_rated_movies_idx)
    # Predict the rating of movie p.
    pred_p = calc_pred(similarity_values, user_rated_movies_r)
    unseen_movies_predictions.append(pred_p)
    
    print('{} / {}'.format(n, len(user_unrated_movies_idx)))
    n+=1

unseen_movies_predictions = np.array(unseen_movies_predictions)
#%%
# Show the N most relevant movies for user.
# Get index values of n largest elements in the vector
def get_n_largest_idx(vec, n=1):
    idxs = (-vec).argsort()[:n]
    return idxs

# user_unrated_movies_idx
# unseen_movies_predictions
N=20
n_largest = get_n_largest_idx(unseen_movies_predictions,N)
print('{} most relevant movies for user {}:'.format(N, user_id+1))
for n in n_largest:
    movie_id = df_movies.get('movieId')[n]
    movie_name = df_movies.get('title')[n]
    print('Pred. rating: {}'.format(unseen_movies_predictions[n]))
    print('(Id : {}), {}\n'.format(movie_id, movie_name))
    