#!/usr/bin/env python
# coding: utf-8

import numpy as np


def calc_euc_distances(x1, x2):
     return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))

def calc_mink_distances(x1, x2):
     return np.power(np.sum((np.array(x1) - np.array(x2)) ** 6), (1/6))

def calc_manh_distances(x1, x2):
     return np.sum(np.abs(np.array(x1) - np.array(x2)))


def prediction(X_train, Y_train, X_test, k, dist_type):

    distances = []
    
    for train_sample in range(len(X_train)):

        if dist_type == "Minkowski":
           dist  = calc_mink_distances(X_test, X_train[train_sample])
        elif dist_type == "Manhattan":
           dist  = calc_manh_distances(X_test, X_train[train_sample])
        else:
            dist = calc_euc_distances(X_test, X_train[train_sample])
        
        if len(distances) < k:
            distances.append((X_train[train_sample], Y_train[train_sample], dist))
            
        elif len([x for x in distances if x[-1] > dist]) > 0:
            index = np.argmax([x[-1] for x in distances])
            distances[index] = (X_train[train_sample], Y_train[train_sample], dist)
    
    labels = [neighbor[1] for neighbor in distances]
    Y_pred = max(set(labels), key=labels.count)
    
    return Y_pred


def knn_model(X_train, X_test, Y_train, k, dist="Euclidean"):
    Y_pred = []
    for test_sample in X_test:
        Y_pred.append(prediction(X_train, Y_train, test_sample, k, dist))
    return np.array(Y_pred) 
    

def calc_accuracy(Y_pred, Y_test):
    errors = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] != Y_test[i]:
            errors += 1

    error_rate = errors/len(Y_pred)
    accuracy = (1-error_rate)*100
    
    return accuracy
