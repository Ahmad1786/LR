
import numpy as np
import matplotlib.pyplot as plt
import random

def closest_number(nums, target):
    closest = nums[0]
    for num in nums:
        if abs(num - target) < abs(closest - target):
            closest = num

    return closest

num_experiments = 1000
N = 100 # number of data points
all_gs = [] #hold the num_experiements best h for each experiment

def target_func(X, Y, m, b):
    if Y > m*X + b:
        return 1
    elif Y < m*X + b:
        return -1

s = 0
for i in range(num_experiments):   
    # create one line
    x1, y1 = np.random.uniform(-1, 1, size=2)
    x2, y2 = np.random.uniform(-1, 1, size=2)
    m = (y2 - y1) / (x2 - x1)
    b = y2 - m*x2
    x_values = np.linspace(-1, 1, 100)
    y_values = m * x_values + b

    random_points = np.random.uniform(-1, 1, size=(N, 2))
    big_X = []
    big_Y = []
    for point in random_points:
        big_X.append([1, point[0], point[1]])
        big_Y.append(target_func(point[0], point[1], m, b))
    
    big_X = np.array(big_X)
    X_T = np.transpose(big_X)
    big_Y = np.array(big_Y)

    psuedo_inverse = np.dot(np.linalg.inv(np.dot(X_T, big_X)), X_T)  
    best_weight = np.dot(psuedo_inverse, big_Y)  
    squared_norm = np.dot(np.dot(big_X, best_weight) - big_Y, np.dot(big_X, best_weight) - big_Y)   
    formula_in_sample_error = squared_norm / N
    
    misclassified_points = 0
    random_points_out = np.random.uniform(-1, 1, size=(1000, 2))
    for point in random_points_out:
        x_vec = [1, point[0], point[1]]
        experimental_sign = np.sign(np.dot(best_weight, x_vec))
        real_sign = target_func(point[0], point[1], m, b)
        if experimental_sign != real_sign:
            misclassified_points += 1
    
