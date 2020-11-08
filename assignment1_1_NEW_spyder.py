import pandas as pd

root = '/home/tuomas/Python/DATA.ML.360/ml-latest-small/'

df_movies = pd.read_csv(root+'movies.csv', usecols=['movieId', 'title'],
                        dtype={'movieId':'int32', 'title':'str'})

df_ratings = pd.read_csv(root+'ratings.csv', usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId':'int32', 'movieId':'int32', 'rating':'float32'})

#%%
from scipy.sparse import csr_matrix
# Pivot ratings into movie features
# THIS CONSIST ONLY RATED MOVIES, NOT ALL MOVIES. 
# MODIFY THIS TO KEEP TRACK ALL MOVIES OR FIND A WAY TO POINT TO COLUMN VALUES:
    # movieId  1       2       3       4       ...  193583  193585  193587  193609
df_movie_features = df_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
# Create sparse feature matrix out of dataframe
mat_movie_ratings = csr_matrix(df_movie_features.values)

#%%
columns = df_movie_features.columns
#%%
# Calculate user rating means
import numpy as np

rating_sums = np.asarray(mat_movie_ratings.sum(axis=1)).ravel()
counts = np.diff(mat_movie_ratings.indptr)
user_means = (rating_sums / counts).ravel()

#%%
# Calculate similarities between users 
import numpy as np
import time
from numpy.linalg import norm

def pearson_correlation(u1_vec, u1_mean, u2_vec, u2_mean):
    u1_meanvec = np.ones(u1_ratings.shape[0]) * u1_mean
    u1_diffvec = u1_vec - u1_meanvec
    
    u2_meanvec = np.ones(u2_ratings.shape[0]) * u2_mean
    u2_diffvec = u2_vec - u2_meanvec
    
    return (u1_diffvec@u2_diffvec) / (norm(u1_diffvec) * norm(u2_diffvec))
   
normalize = True
start = time.time()
sim_matrix = np.zeros((609 + 1, 609 + 1))
cases_matrix = np.zeros((609 + 1, 609 + 1))
not_found = -1 if normalize else 0
for uid1 in range(610):
    for uid2 in range(uid1, 610):
        # Comparing user to itself.
        if(uid1==uid2): 
            sim_matrix[uid1, uid2] = not_found
            cases_matrix[uid1, uid2] = 0
            continue
        
        u1_ratings = mat_movie_ratings.getrow(uid1).toarray().ravel()
        u2_ratings = mat_movie_ratings.getrow(uid2).toarray().ravel()
        common_ratings = np.nonzero(u1_ratings * u2_ratings)
        u1_ratings = u1_ratings[common_ratings]
        u2_ratings = u2_ratings[common_ratings]
        
        #print(uid1, uid2)
        pc = pearson_correlation(u1_ratings, user_means[uid1], u2_ratings, user_means[uid2])
        if(np.isnan(pc)):
            pc = not_found
        sim_matrix[uid1, uid2] = pc
        cases_matrix[uid1, uid2] = u1_ratings.shape[0]
        
    print("User {} ready".format(uid1))
print("Run time = {} min".format(round((time.time() - start) / 60.0, 1)))
        
# Normailze sim matrix
if(normalize):
    sim_matrix += 1
    sim_matrix *= 0.5
        
# Copy upper triangle to lower triangle
ltr_idx = np.tril_indices(sim_matrix.shape[0], -1)
sim_matrix[ltr_idx] = sim_matrix.T[ltr_idx]
cases_matrix[ltr_idx] = cases_matrix.T[ltr_idx]

#%%
# OPTIONAL: Set sim_matrix values to 0 if similarity score
# was calculated from too few samples..
filt = cases_matrix < 1 # This setting makes no effect
sim_matrix[filt] = 0        

#%%
# Select a user from the dataset, and for this user, show the N most similar users
def get_n_largest_idx(vec, n=1):
    idxs = (-vec).argsort()[:n]
    return idxs

user_id = 0
N = 10
sim_users = get_n_largest_idx(sim_matrix[user_id,:],N)
print("{} most similar users for user {}".format(N, user_id+1))
for n in range(N):
    print('User {}, similarity score: {}, cases: {}'.format(sim_users[n]+1, sim_matrix[user_id,sim_users[n]],
                                                            cases_matrix[user_id,sim_users[n]]))

#%%
# Predict rating for all unrated movies for user

def predict_rating(uid, mid):
    ratings = mat_movie_ratings.getcol(mid).toarray().ravel()
    rated_user_ids = np.nonzero(ratings)
    rated_user_ratings = ratings[rated_user_ids]
    
    # Get the mean ratings of rated users
    rated_user_means = user_means[rated_user_ids]
    
    # Calculate rating prediction
    mean_diff = rated_user_ratings - rated_user_means
    sim_vector = sim_matrix[uid, rated_user_ids]
    
    return user_means[uid] + (sim_vector@mean_diff) / sim_vector.sum()

def predict_unrated(uid):
    all_ratings = mat_movie_ratings.getrow(uid).toarray().ravel()
    unrated_movies = np.where(all_ratings == 0)[0]
    # Predict scores for all unseen movies
    pred_ratings = []
    for movie_id in unrated_movies:
        pred = predict_rating(uid, movie_id)
        pred_ratings.append(pred)
        
    return np.array(pred_ratings).ravel(), np.array(unrated_movies).astype('int')
        
        
user_id = 0
pred_ratings, movie_idxs = predict_unrated(user_id)
#%%
# Print top suggestions for user
N=5
top_filter = get_n_largest_idx(pred_ratings, N)
pred_ratings_top = pred_ratings[top_filter]
movie_idxs_top = movie_idxs[top_filter]

all_movie_ids = df_movie_features.columns
print('List of {} most relevant movies for user {}:'.format(N, user_id+1))
for pr,mi in zip(pred_ratings_top, movie_idxs_top):
    movie_id = all_movie_ids[mi]
    movie_name = df_movies.loc[df_movies['movieId']==movie_id].get('title').values[0]
    print('Pred. rating: {}'.format(pr))
    print('(Id : {}), {}\n'.format(movie_id, movie_name))
