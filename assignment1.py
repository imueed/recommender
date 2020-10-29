import numpy as np
import time

# Pearson-correlation score
# Params : Two user-id's and corresponding mean rating values
# Return : Pearson correlation, e.g similarity score
def pearson_correlation(u1_ratings, u1_rmean, u2_ratings, u2_rmean):
    # Calculate numerator of Pearson correlation formula.
    # Vector algebra is used.
    u1_rmean_vec = np.ones(u1_ratings.shape[0]) * u1_rmean
    u1_diff_vec = u1_ratings - u1_rmean_vec
    
    u2_rmean_vec = np.ones(u2_ratings.shape[0]) * u2_rmean
    u2_diff_vec = u2_ratings - u2_rmean_vec
    
    numerator = np.dot(u1_diff_vec, u2_diff_vec)
    
    # Calculate denominator of Pearson correlation formula.
    # Vector algebra is used.
    u1_diff_sqrt = np.sqrt(np.dot(u1_diff_vec, u1_diff_vec))
    u2_diff_sqrt = np.sqrt(np.dot(u2_diff_vec, u2_diff_vec))
    
    denominator = u1_diff_sqrt * u2_diff_sqrt
    
    return numerator / denominator

# Get users who have rated the movie item_id.
# Params : ID-value of movie
# Return : List of user-id's and respective rating scores
def get_rated_users(item_id):
    data = ratings[ratings[:,1]==item_id]
    return np.delete(data, 1, axis=1)

# Get movies watched by both users
# Params : ID-values for both users
# Return : List of movie-id's
def get_similar_movies(uid1, uid2):
    user1_movies = ratings[np.where(ratings[:,0] == uid1)][:,1]
    user2_movies = ratings[np.where(ratings[:,0] == uid2)][:,1]
    
    return np.intersect1d(user1_movies, user2_movies)

# Get rating data for specific user
# Params : User-id and attribute indexes
# Return : User-id-specific data
def get_userdata(uid, first_attr=0, last_attr=3):
    data = ratings[np.where(ratings[:,0] == uid)][:, first_attr:last_attr]
    if (last_attr - first_attr) == 1:
        data = data.T
    return data

root = '/home/tuomas/Python/DATA.ML.360/ml-latest-small/'

ratings = np.genfromtxt(root+'ratings.csv', delimiter=',')
ratings = np.delete(ratings[1:], 3, axis=1)
# Start user indexes at 0 instead at 1
ratings[:,0] -= 1

first_uid = int(ratings[0,0])
last_uid = int(ratings[-1,0])
user_ids = np.arange(first_uid, last_uid+1)

# NOTE: First element stays at zero.
user_means = np.zeros(last_uid+1)
for uid in user_ids:
    user_means[uid] = np.mean(get_userdata(uid, first_attr=2, last_attr=3))

# Matrix to store similarity measures between users.
# Note, that due to symmetry, the created matrix is upper triangle.
start = time.time()
sim_matrix = np.zeros((last_uid + 1, last_uid + 1))
for uid1 in user_ids:
    for uid2 in user_ids[uid1:]:
        # Set correlation coefficient to minimum when comparing user to itself.
        if(uid1==uid2): 
            sim_matrix[uid1, uid2] = -1.0
            continue
        
        sim_movies = get_similar_movies(uid1, uid2)
        
        # ui_data[i] consist values of [user_id, movie_id, rating]
        u1_data = get_userdata(uid1)
        u1_filter = np.in1d(u1_data[:,1], sim_movies)
        u1_data = u1_data[u1_filter]
        u1_ratings = u1_data[:,2]
        
        u2_data = get_userdata(uid2)
        u2_filter = np.in1d(u2_data[:,1], sim_movies)
        u2_data = u2_data[u2_filter]
        u2_ratings = u2_data[:,2]
        
        pc = pearson_correlation(u1_ratings, user_means[uid1], u2_ratings, user_means[uid2])
        if(np.isnan(pc)):
            # print(pc)
            pc = -1.0
            
        sim_matrix[uid1, uid2] = pc
        
    print("User {} ready".format(uid1))
    
print("Run time = {} min".format(round((time.time() - start) / 60.0, 1)))

#%%
# Normailze sim matrix
sim_matrix += 1
sim_matrix *= 0.5

#%%
# Copy upper triangle to lower triangle
ltr_idx = np.tril_indices(sim_matrix.shape[0], -1)
sim_matrix[ltr_idx] = sim_matrix.T[ltr_idx]

#%%
""" Select a user from the dataset, and for this user, show the 10 most similar users """
# Returns indecies of n largest elements ( Correct results )
def get_n_largest2(arr, n=1):
    arr = np.argpartition( -arr, n)[:n]
    arr.sort()
    return arr

def similar_users(userid, n=1):
    indices = (-sim_matrix[userid]).argsort()[:n]
    indices.sort()
    return indices

user_id = 1
N = 5
# sim_users = get_similar_users(user_id,N)
sim_users = get_n_largest2(sim_matrix[user_id,:],N)
#sim_users = similar_users(user_id ,N)

print("{} most similar users for user {}".format(N, user_id+1))
for n in range(N):
    print('User {}, similarity score: {}'.format(sim_users[n]+1, sim_matrix[user_id,sim_users[n]]))

#%%
""" Show the 20 most relevant movies """
import pandas as pd

# Calculate rating prediction for unseen movie for specific user
# Params : User-id and movie-id of unseen movie
# Return : Rating prediction for unseen movie
def predict_movie_rating1(uid, item_id):
    # Get the list of users who have rated the movie itemid
    rated_users = get_rated_users(item_id)
    rated_user_ids = rated_users[:,0].astype('int16')
    rated_user_ratings = rated_users[:,1]
    # Get the mean ratings of rated_users
    rated_user_means = user_means[rated_user_ids]
    
    # Calculate rating prediction
    # @ == dot product
    mean_diff = rated_user_ratings - rated_user_means
    sim_vector = sim_matrix[uid, rated_user_ids]
    numerator = sim_vector @ mean_diff
    
    # print(user_means[uid], numerator, sim_vector.sum())
    return user_means[uid] + (numerator / sim_vector.sum())
    #return rated_user_means + (numerator / sim_vector.sum())

def get_unseen_movies(uid):
    seen_movies = get_userdata(uid, 1, 2).ravel()
    unseen_movies = np.setdiff1d(movies[:,0], seen_movies)
    return unseen_movies

def get_relevant_movies(uid, N=1):
    unseen_movies = get_unseen_movies(uid)
    pred_ratings = []
    # Predict scores for all unseen movies
    for movie in unseen_movies:
        pred = predict_movie_rating1(uid, movie)
        pred_ratings.append(pred)
      
    most_relevant_ids = get_n_largest2(np.array(pred_ratings), N)
    print(most_relevant_ids.shape)
    most_relevant_scores = np.array(pred_ratings)[most_relevant_ids]
    print(most_relevant_scores)
    
    return np.concatenate((most_relevant_ids, most_relevant_scores)).reshape((-1,2), order='F')

movies = pd.read_csv(root+'movies.csv', sep=',').values

user_id = 1
N=5
most_relevant_movies = get_relevant_movies(user_id)
#[2662 4586 5197 6045 9230]
#%%
print('{} most relevant movies for user {}'.format(N, user_id+1))
for mrm in most_relevant_movies:
    movie_id = int(mrm[0])
    print('MovieID {}:'.format(movies[movie_id,0]))
    #print('MovieID2 {}:'.format(movies[np.where(movies[:,0]==movie_id)][0]))
    print('{}'.format(movies[movie_id,1]))
    print('Predicted score = {}\n'.format(mrm[1]))

#%%
""" Item based collaborative filtering """
from numpy.linalg import norm

# Cosine difference (angle) between two vectors sim(a, b)
# Larger is better
# -1 : vectors are opposite
# 0 : vectors are orthogonal
# 1 : vectors are same
def cosine_diff(a, b):
    return (a@b) / (norm(a)*norm(b))

def get_movie_data(movie_id):
    return ratings[np.where(ratings[:,1] == movie_id)]

def filter_data(movie_ratings1, movie_ratings2):
    filt = np.intersect1d(movie_ratings1[:,0], movie_ratings2[:,0])

    filt1 = np.in1d(movie_ratings1[:,0], filt)
    filt2 = np.in1d(movie_ratings2[:,0], filt)
    
    return movie_ratings1[filt1], movie_ratings2[filt2]


movie_id = 1
mr1 = get_movie_data(1)
mr2 = get_movie_data(2)

mr1, mr2 = filter_data(mr1, mr2)

mr1 = mr1[:,2]
mr2 = mr2[:,2]
sim = cosine_diff(mr1, mr2)

#%%
# Calculates the similarity vector and other components needed
def get_prediction_components(uid, mid):
    # Check formula for this one
    p_data = get_movie_data(mid)
    # Get the list of movies that user uid has rated.
    u_movie_data = get_userdata(uid)
    u_movie_ids = u_movie_data[:,1].astype('int') # ratedItem(u)
    u_movie_ratings = u_movie_data[:,2] #r_ui
    
    # Get all movie ratings indexed by u_movie_ids
    ratings_data = []
    for umi in u_movie_ids:
        r_data = get_movie_data(umi)
        ratings_data.append(r_data)
        
    ratings_data = np.array(ratings_data, dtype='object')
    
    filtered_data_p = []
    filtered_data_r = []
    movie_ids = []
    n=0
    for rd in ratings_data:
        fpdata, frd = filter_data(p_data, rd)
        filtered_data_p.append(fpdata[:,2])
        filtered_data_r.append(frd[:,2])
        
        movie_id = np.nan if frd.shape[0]==0 else frd[0,1]
        movie_ids.append(movie_id)
        n+=1
    
    sim_vector = []
    for fdp, fdr in zip(filtered_data_p, filtered_data_r):
        cos_diff = cosine_diff(fdp, fdr)
        sim_vector.append(cos_diff)
    
    sim_vector = np.array(sim_vector)
    print(sim_vector.shape, u_movie_ratings.shape, len(movie_ids))
    print('ok')  
    # Return all components (slide 22)
    return sim_vector, u_movie_ratings, np.array(movie_ids) 
    #return np.array(filtered_data_p, dtype='object'), np.array(filtered_data_r, dtype='object'), movie_ids

def predict_movie_rating2(sim_vec, r_vec):
    numerator = sim_vec@r_vec
    denominator = np.sum(sim_vec)
    return numerator / denominator

uid=0
u_unseen = get_unseen_movies(uid)[0]
sim_vec, umr_vec, mids = predict_movie_rating2(uid, u_unseen)

















