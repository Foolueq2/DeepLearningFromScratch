# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        #当你使用 enumerate(X) 时，它会返回一个包含 (index, value) 元组的迭代器：
        #index 是当前元素在 X 中的索引（从 0 开始）。
        #value 是 X 中对应的元素。
        # X = np.array([[1, 2, 3],
        #               [4, 5, 6],
        #               [7, 8, 9]])
        # for idx, row in enumerate(X):
        #     print(f"Index: {idx}, Row: {row}")
        #Index: 0, Row: [1 2 3]
        # Index: 1, Row: [4 5 6]
        # Index: 2, Row: [7 8 9]
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
# np.nditer 是一个强大的迭代器，支持对多维数组的高效遍历。
# flags=['multi_index'] 使得可以使用多维索引（即 multi_index）。
# op_flags=['readwrite'] 表示该数组可以被读取和修改。
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad
