"""
src/evaluate.py

Purpose: This module contains functions to evaluate model performance using various metrics such as: 
1. Mean Square Error (MSE), 
2. Root Mean Square Error (RMSE), 
3. Mean Absolute Error (MAE), 
4. Mean Absolute Percentage Error (MAPE).
"""

import numpy as np

def mse(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_all(y_true, y_pred):
    """Returns all metrics in a dictionary"""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }

if __name__ == "__main__":
    # Example usage
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    
    results = evaluate_all(y_true, y_pred)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")