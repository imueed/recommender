#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:24:24 2020

@author: mueed
"""


##%% New cell

import numpy as np

root = '/Users/mueed/Downloads/ml-latest-small/'

def pearson_correlation(user1, user2):
    
    return 0

 

def get_similar_movies(uid1, uid2):
    user1_movies = ratings[np.where(ratings[:,0] == uid1)][:,1]
    user2_movies = ratings[np.where(ratings[:,0] == uid2)][:,1]
    
    return np.intersect1d(user1_movies, user2_movies)
 

ratings = np.genfromtxt(root+'ratings.csv', delimiter=',')

 

ratings = np.delete(ratings[1:], 3, axis=1)

 

first_uid = ratings[0,0]
last_uid = ratings[-1,0]

 

user_means = np.zeros(int(last_uid))
for uid in range(int(first_uid), int(last_uid)):
    user_means[uid] = np.mean(ratings[np.where(ratings[:,0] == uid)][:,2])
    # print("ID={}, mean={}".format(uid, user_means[uid]))

 

similarity_matrix = np.zeros((last_uid, last_uid))
for uid1 in range(int(first_uid), int(last_uid)):
    for uid2 in range(int(first_uid), int(last_uid)):
        if(uid1==uid2): continue
        
        similar_movies = get_similar_movies(uid1, uid2)
        
        u1_ratings = ratings[np.where(ratings[:,0] == uid1)]
        u1_ratings_ = []
        for ur in u1_ratings:
            if(ur[1] in similar_movies):
                u1_ratings_.append(ur[2])

 

        u1_similar_movie_ratings = np.array(u1_ratings_)
        
        
        u2_ratings = ratings[np.where(ratings[:,0] == uid2)]
        u2_ratings_ = []
        for ur in u2_ratings:
            if(ur[1] in similar_movies):
                u2_ratings_.append(ur[2])
        
        u2_similar_movie_ratings = np.array(u2_ratings_)
        
        # TODO :  Calculate similarity matrix values based on 
        # u1_similar_movie_ratings and u2_similar_movie_ratings and corresponding means
        
        print(u1_ratings.shape, u2_ratings.shape)
        