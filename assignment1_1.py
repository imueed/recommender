import numpy as np
import time
import pandas as pd
from numpy.linalg import norm

# Pearson-correlation score
# Params : Two user rating vectors and corresponding mean rating values
# Return : Pearson correlation, e.g similarity score
def pearson_correlation(u1_ratings, u1_rmean, u2_ratings, u2_rmean):
    # Calculate numerator of Pearson correlation formula.
    u1_rmean_vec = np.ones(u1_ratings.shape[0]) * u1_rmean
    u1_diff_vec = u1_ratings - u1_rmean_vec
    
    u2_rmean_vec = np.ones(u2_ratings.shape[0]) * u2_rmean
    u2_diff_vec = u2_ratings - u2_rmean_vec
    
    numerator = np.dot(u1_diff_vec, u2_diff_vec)
    
    # Calculate denominator of Pearson correlation formula.
    denominator = norm(u1_diff_vec) * norm(u2_diff_vec)
    
    return numerator / denominator

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

# Return matrix which stores similarity measures between users.
def calculate_sim_matrix(normalize=True):
    start = time.time()
    sim_matrix = np.zeros((last_uid + 1, last_uid + 1))
    cases_matrix = np.zeros((last_uid + 1, last_uid + 1))
    not_found = -1 if normalize else 0
    # Note, that due to symmetry, the matrix created inside loop is upper triangle.
    for uid1 in user_ids:
        for uid2 in user_ids[uid1:]:
            # Comparing user to itself.
            if(uid1==uid2): 
                sim_matrix[uid1, uid2] = not_found
                cases_matrix[uid1, uid2] = 0
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
            # If similarity score cannot be calculated
            # (set of items, rated both by u1 and u2 is empty)
            if(np.isnan(pc)):
                pc = not_found
            sim_matrix[uid1, uid2] = pc
            cases_matrix[uid1, uid2] = u1_ratings.shape[0]
        
        print("User {} ready".format(uid1))
        
    print("Run time = {} min".format(round((time.time() - start) / 60.0, 1)))
    
    # Normailze similarity matrix
    if(normalize):
        sim_matrix += 1
        sim_matrix *= 0.5
    
    # Copy upper triangle to lower triangle
    ltr_idx = np.tril_indices(sim_matrix.shape[0], -1)
    sim_matrix[ltr_idx] = sim_matrix.T[ltr_idx]
    cases_matrix[ltr_idx] = cases_matrix.T[ltr_idx]
    
    return sim_matrix, cases_matrix

# Get index values of n largest elements in the vector
def get_n_largest_idx(vec, n=1):
    idxs = (-vec).argsort()[:n]
    return idxs

def get_movie_name(mid):
    rowidx = np.where(movies[:,0]==mid)[0]
    return movies[rowidx].ravel()[1]

def print_user_top_ratings(uid, N=1):
    seen_movie_data = get_userdata(uid, 1, 3)
    sorted_movie_data = seen_movie_data[np.argsort(seen_movie_data[:, 1])]
    sorted_movie_data = sorted_movie_data[::-1]
    
    seen_movie_ids = sorted_movie_data[:,0][:N]
    seen_movie_ratings = sorted_movie_data[:,1][:N]

    print("Top {} list of movies that user {} has rated:".format(N, uid+1))
    for mid, r in zip(seen_movie_ids, seen_movie_ratings):
        name = get_movie_name(mid)
        print('#{} (Id: {}) {}'.format(r, int(mid), name))  

# Get users who have rated the movie item_id.
# Params : ID-value of movie
# Return : List of user-id's and respective rating scores
def get_rated_users(item_id):
    data = ratings[ratings[:,1]==item_id]
    return np.delete(data, 1, axis=1)

# Calculate rating prediction for unseen movie for specific user
# Params : User-id and movie-id of unseen movie
# Return : Rating prediction for unseen movie
def predict_movie_rating(uid, item_id):
    # Get the list of users who have rated the movie itemid
    rated_users = get_rated_users(item_id)
    rated_user_ids = rated_users[:,0].astype('int')
    rated_user_ratings = rated_users[:,1]
    # Get the mean ratings of rated_users
    rated_user_means = user_means[rated_user_ids]
    
    # Calculate rating prediction
    mean_diff = rated_user_ratings - rated_user_means
    sim_vector = sim_matrix[uid, rated_user_ids] 
    
    numerator = sim_vector @ mean_diff
    
    # print(user_means[uid], numerator, sim_vector.sum())
    return user_means[uid] + (numerator / sim_vector.sum())

def get_unseen_movies(uid):
    seen_movies = get_userdata(uid, 1, 2).ravel()
    unseen_movies = np.setdiff1d(movies[:,0], seen_movies)
    return unseen_movies

def get_relevant_movies(uid, N=1):
    unseen_movies = get_unseen_movies(uid).astype('int')
    pred_ratings = []
    # Predict scores for all unseen movies
    for movie_id in unseen_movies:
        pred = predict_movie_rating(uid, movie_id)
        pred_ratings.append(pred)
     
    most_relevant_filter = get_n_largest_idx(np.array(pred_ratings), N)
    most_relevant_ids = unseen_movies[most_relevant_filter]
    most_relevant_scores = np.array(pred_ratings)[most_relevant_filter]
    
    return np.concatenate((most_relevant_ids, most_relevant_scores)).reshape((-1,2), order='F')
        

# CHANGE DIRECTORY PATH HERE TO THE ASSIGNMENT FILES 
#root = '/home/tuomas/Python/DATA.ML.360/ml-latest-small/'
root = '/Users/mueed/Documents/GitHub/recommender/ml-latest-small/'

movies = pd.read_csv(root+'movies.csv', sep=',').values

ratings = np.genfromtxt(root+'ratings.csv', delimiter=',')
ratings = np.delete(ratings[1:], 3, axis=1)
# Start user id's at 0 instead at 1
ratings[:,0] -= 1

first_uid = int(ratings[0,0])
last_uid = int(ratings[-1,0])
user_ids = np.arange(first_uid, last_uid+1)

# Calculate user rating means
user_means = np.zeros(last_uid+1)
for uid in user_ids:
    user_means[uid] = np.mean(get_userdata(uid, first_attr=2, last_attr=3))

# Calculate similarities between users (Takes about 2min to calculate)
sim_matrix, cases_matrix = calculate_sim_matrix()

# OPTIONAL: Set sim_matrix values to 0 if similarity score
# was calculated from too few samples.
filt = cases_matrix < 1 # This setting makes no effect
sim_matrix[filt] = 0

# Select a user from the dataset, and for this user, show the N most similar users
# CHANGE THE USER_ID AND N VALUE HERE.
user_id = 0
N = 10
sim_users = get_n_largest_idx(sim_matrix[user_id,:],N)
print()
print("{} most similar users for user {}".format(N, user_id+1))
for n in range(N):
    print('User {}, similarity score: {}, cases: {}'.format(sim_users[n]+1, sim_matrix[user_id,sim_users[n]],
                                                            cases_matrix[user_id,sim_users[n]]))

# Print the movies rated by specific user 
print()
print_user_top_ratings(user_id, 10)

# Show the N most relevant movies

N = 20
most_relevant_movies = get_relevant_movies(user_id, N)
print()
print('List of {} most relevant movies for user {}:'.format(N, user_id+1))
for mrm in most_relevant_movies:
    movie_id = int(mrm[0])
    movie_name = get_movie_name(movie_id)
    r = mrm[1]
    
    print('Pred. rating: {}'.format(r))
    print('(Id : {}), {}\n'.format(movie_id, movie_name))



