
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
N = 10 # number of data points
all_gs = [] #hold the num_experiements best h for each experiment

def target_func(X, Y, m, b):
    if Y > m*X + b:
        return 1
    elif Y < m*X + b:
        return -1

def apply_LR(random_points):
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

    return best_weight

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
    weight_vector = apply_LR(random_points)

    solved = False
    iterations = 0
    while not solved:
        missclassed_points = []
        iterations += 1
        for arr in random_points:
            x, y = arr[0], arr[1]
            x_vector = [1, x, y]
            x_vector = np.array(x_vector)
            sign = np.sign(np.dot(weight_vector, x_vector))
            correct_sign = target_func(x, y, m, b)
            if sign != correct_sign:
                missclassed_points.append((x, y))

        if missclassed_points:
            missed_point = random.choice(missclassed_points)
            weight_vector = weight_vector + target_func(missed_point[0], missed_point[1], m, b)*np.array([1, missed_point[0], missed_point[1]])
        else:
            solved = True
            
            s += iterations
    
print(s/num_experiments)           

        
    
    


