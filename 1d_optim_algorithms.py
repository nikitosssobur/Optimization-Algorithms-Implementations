import numpy as np
import matplotlib.pyplot as plt
import math
import autograd.numpy as anp
from autograd import grad 


'''
This file contains some 1D-optimization methods implementations.
Some of these methods are gradient-based some of them require computing
just the values of optimizable functions (without gradients)  
'''


get_golden_ratio_roots = lambda: (math.sqrt(5) - 1) / 2, (math.sqrt(5) + 1) / 2



def optimize(a, b, func, epsilon):
    step_counter = 0
    while b - a >= epsilon:
        left_center_point, right_center_point = np.linspace(a, b, 4)[1:-1]   #c, d  points definition
        if func(left_center_point) >= func(right_center_point):
            a = left_center_point
        else:
            b = right_center_point
        step_counter += 1

    return a, b, step_counter


'''
def dichotomy_method(a, b, func, epsilon):
    step_counter = 0
    center_point = (a + b) / 2
    add_point = (a + center_point) / 2
    while b - a >= epsilon:
        if func(add_point) < func(center_point):
            b, center_point, add_point = center_point, add_point, (a + add_point) / 2
        else:
            a, add_point, center_point = add_point, center_point, (center_point + b) / 2
        step_counter += 1

    return a, b, step_counter
'''


def dichotomy_method(a, b, func, epsilon):
    step_counter, center_point = 0, (a + b) / 2
    add_point = (a + center_point) / 2
    old_add_point = add_point
    while b - a >= epsilon:
        if func(add_point) < func(center_point): b, center_point = center_point, add_point
        else:
            old_add_point, add_point = add_point, (center_point + b) / 2
            if func(center_point) < func(add_point):  a,  b = old_add_point, add_point
            else: a, center_point = center_point, add_point
        
        old_add_point, add_point = add_point, (a + center_point) / 2
        step_counter += 1

    return a, b, step_counter



def divide_segment(start_point_coords, end_point_coords, ratio: float):
    #Function divides a segment in a given ratio
    x_output_coord = ((start_point_coords[0] + ratio * end_point_coords[0]) / (1 + ratio))
    y_output_coord = ((start_point_coords[1] + ratio * end_point_coords[1]) / (1 + ratio))
    return (x_output_coord, y_output_coord)


def golden_ratio_method(a: float, b: float, func, epsilon: float):
    left_inner_point = divide_segment((a, 0), (b, 0), min(get_golden_ratio_roots()))[0]
    right_inner_point = a + b - left_inner_point
    left_in_func_value, right_in_func_value = func(left_inner_point), func(right_inner_point)
    step_counter = 0
    while b - a >= epsilon:
        if left_in_func_value > right_in_func_value:
            a, left_inner_point = left_inner_point, right_inner_point
            right_inner_point = divide_segment((a, 0), (b, 0), max(get_golden_ratio_roots()))[0]
            left_in_func_value, right_in_func_value = right_in_func_value, func(right_inner_point)
        else:
            b, right_inner_point = right_inner_point, left_inner_point
            left_inner_point = divide_segment((a, 0), (b, 0), min(get_golden_ratio_roots()))[0]
            right_in_func_value, left_in_func_value = left_in_func_value, func(left_inner_point) 
        step_counter += 1
        
    return a, b, step_counter


def gradient_descent(x0, eps, learning_rate, func):
    '''
    Function returns the dictionary in the following format: 
        {'x_est': x estimation, 'func_value': func_value at the found point, 
        'iter_num': number of iterations, 'func_call_num': number of calls of functions}
    '''
    grad_func = grad(func)    
    x_old, x_new = float(x0), x0 + abs(eps) + 1
    iter_num = 0
    while abs(func(x_new) - func(x_old)) >= eps:
        x_old = x_new
        x_new = x_old - learning_rate * grad_func(x_old)
        iter_num += 1
        
    return {'x_est': x_new, 'func_value': func(x_new), 'iter_num': iter_num, 'func_call_num': iter_num + 1}


def heavy_ball_method(x0, eps, learning_rate, momentum_coeff, func):
    '''
    x_new -- x{n + 1}, x_old -- x{n}, x_2old -- x{n - 1}
    '''
    grad_func = grad(func)
    x_old, x_new = float(x0), x0 + abs(eps) + 1
    iter_num = 0
    while abs(func(x_new) - func(x_old)) >= eps:
        if iter_num < 1:
            x_new = x_old - learning_rate * grad_func(x_old)
        else:
            x_2old, x_old = x_old, x_new
            x_new = x_old - learning_rate * grad_func(x_old) + momentum_coeff * (x_old - x_2old)
        
        iter_num += 1
        
    return {'x_est': x_new, 'func_value': func(x_new), 'iter_num': iter_num, 'func_call_num': iter_num + 1}



def newton_raphson_method(x0: float, eps: float, func) -> dict:
    x_old, x_new = float(x0), x0 + 10
    iter_num = 0
    grad_func = grad(func)
    while abs(func(x_old) - func(x_new)) >= eps:
        x_old = x_old if iter_num == 0 else x_new 
        x_new = x_old - func(x_old) / grad_func(x_old)
        iter_num += 1    
    
    return {'x_est': x_new, 'func_value': func(x_new), 'iter_num': iter_num, 'func_call_num': iter_num + 1}


def secant_method(x0: float, x1: float, eps: float, func):
    x_old, x_new = float(x0), x1
    iter_num = 0
    while abs(func(x_old) - func(x_new)) >= eps: 
        x_old = x_old if iter_num == 0 else x_new
        denominator = (func(x_old) - func(x_new)) / (x_old - x_new) 
        x_new = x_new - func(x_new) / denominator
    
    return {'x_est': x_new, 'func_value': func(x_new), 'iter_num': iter_num, 'func_call_num': iter_num + 1}


'''
Some experiments with test functions
'''

if __name__ == "__main__":
    x_squared = lambda x: x ** 2
    function = lambda x: 0.5 - x * anp.exp(-x**2)
    epsilon = 0.00001
    func2 = lambda x: x_squared(x_squared(x))

    res1 = gradient_descent(0.8, 0.0001, 0.01, function)
    res2 = heavy_ball_method(0.8, 0.0001, 0.01, 0.05, function)
    res3 = newton_raphson_method(0.8, 0.000001, function)

    print(res1)
    print(res2)
    print(res3)