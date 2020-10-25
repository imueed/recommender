import numpy as np
import time

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

def get_similar_movies(uid1, uid2):
    user1_movies = ratings[np.where(ratings[:,0] == uid1)][:,1]
    user2_movies = ratings[np.where(ratings[:,0] == uid2)][:,1]
    
    return np.intersect1d(user1_movies, user2_movies)

def get_userdata(uid, first_attr=0, last_attr=3):
    data = ratings[np.where(ratings[:,0] == uid)][:, first_attr:last_attr]
    if (last_attr - first_attr) == 1:
        data = data.T
    return data

root = '/home/tuomas/Python/DATA.ML.360/ml-latest-small/'

ratings = np.genfromtxt(root+'ratings.csv', delimiter=',')
ratings = np.delete(ratings[1:], 3, axis=1)

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
    for uid2 in user_ids[uid1-1:]:
        # Set correlation coefficient to minimum when comparing user to itself.
        if(uid1==uid2): 
            sim_matrix[uid1, uid2] = -666.0
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
        
        sim_matrix[uid1, uid2] = pearson_correlation(u1_ratings, user_means[uid1],
                                                     u2_ratings, user_means[uid2])
        
    print("User {} ready".format(uid1))
    
print("Run time = {} min".format(round((time.time() - start) / 60.0, 1)))

# Copy upper triangle to lower triangle
ltr_idx = np.tril_indices(sim_matrix.shape[0], -1)
sim_matrix[ltr_idx] = sim_matrix.T[ltr_idx]