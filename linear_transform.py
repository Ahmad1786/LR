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
s = 0
s2 = 0
for _ in range(num_experiments):
    random_points = np.random.uniform(-1, 1, size=(N, 2))
    classification = []
    for point in random_points:
        class_type = target_func(point[0], point[1])
        classification.append(class_type)
    classification = np.array(classification)
    
    weight_vector = apply_LRwithTransformation(random_points, classification)

    missclassed_points = 0
    for arr in random_points:
        x_vector = [1, arr[0], arr[1]]
        x_vector = np.array(x_vector)
        z_vector = [1, arr[0], arr[1], arr[0]*arr[1], arr[0]*arr[0], arr[1]*arr[1]]
        z_vector = np.array(z_vector)
        sign = np.sign(np.dot(weight_vector, z_vector))
        correct_sign = target_func(arr[0], arr[1])
        if sign != correct_sign:
            missclassed_points += 1
    insample_error = missclassed_points/N

    s += insample_error

    missclassed_points2 = 0
    random_points2 = np.random.uniform(-1, 1, size=(N, 2))   
    classification2 = []
    for point in random_points2:
        class_type = target_func(point[0], point[1])
        classification2.append(class_type)
    classification2 = np.array(classification2)     
    num_to_flip = int(0.10 * len(classification))
    indices_to_flip = np.random.choice(len(classification2), num_to_flip, replace=False)
    for idx in indices_to_flip:
        classification2[idx] = -1 if classification2[idx] == 1 else 1
    for point, sign in zip(random_points2, classification2):
        z_vector = [1, point[0], point[1], point[0]*point[1], point[0]*point[0], point[1]*point[1]]
        z_vector = np.array(z_vector)
        experimental_sign = np.sign(np.dot(weight_vector, z_vector))
        if experimental_sign != sign:
            missclassed_points2 += 1
    
    s2 += missclassed_points2/N

    
print(f"in sample error: {s/num_experiments}")  
print(f"out of sample error: {s2/num_experiments}") 

