import numpy as np

def relu(x):                    
    return np.maximum(0, x)


def relu_prime(x):              
    return (x > 0).astype(float)


def categorical_cross_entropy(y_true, y_pred):
    eps = 1e-9
    return -np.sum(y_true * np.log(y_pred + eps))



def categorical_cross_entropy_prime(y_true, y_pred):
    eps = 1e-9
    return -y_true / (y_pred + eps)