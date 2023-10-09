import numpy as np
import matplotlib.pyplot as plt
import random

num_experiments = 1000
N = 1000 # number of data points
def target_func(X, Y):
    # X^2 + Y^2 -0.6
    point = X**2 + Y**2 - 0.6
    if point > 0:
        return 1
    if point < 0:
        return -1
    
def apply_LRwithTransformation(random_points, classification):
    big_X = []
    for point in random_points:
        big_X.append([1, point[0], point[1], point[0]*point[1], point[0]*point[0], point[1]*point[1]])
    
    big_X = np.array(big_X)
    X_T = np.transpose(big_X)
    classification = np.array(classification)

    psuedo_inverse = np.dot(np.linalg.inv(np.dot(X_T, big_X)), X_T)  
    best_weight = np.dot(psuedo_inverse, classification)

    return best_weight

def qtypes(letter, x1, x2):
    if letter == 'A':
        return np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1*x1 + 1.5*x2*x2)
    if letter == 'B':
        return np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1*x1 + 15*x2*x2)
    if letter == 'C':
        return np.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 15*x1*x1 + 1.5*x2*x2)
    if letter == 'D':
        return np.sign(-1 - 1.5*x1 + 0.08*x2 + 0.13*x1*x2 + 0.05*x1*x1 + .05*x2*x2)
    if letter == 'E':
        return np.sign(-1 - 0.05*x1 + 0.08*x2 + 1.5*x1*x2 + .15*x1*x1 + .15*x2*x2)

for _ in range(12):
    random_points = np.random.uniform(-1, 1, size=(N, 2))
    classification = []
    for point in random_points:
        class_type = target_func(point[0], point[1])
        classification.append(class_type)
    classification = np.array(classification)

    options = [['A', 0], ['B', 0], ['C', 0], ['D', 0], ['E', 0]]
    weight_vector = apply_LRwithTransformation(random_points, classification)
    for arr in random_points:
        z_vector = [1, arr[0], arr[1], arr[0]*arr[1], arr[0]*arr[0], arr[1]*arr[1]]
        my_sign = np.sign(np.dot(weight_vector, z_vector))
        for el in options:
            if qtypes(el[0], arr[0], arr[1]) == my_sign:
                el[1] += 1
    print(options)
    

