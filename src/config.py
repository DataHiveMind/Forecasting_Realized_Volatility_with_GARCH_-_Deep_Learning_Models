"""
src/config.py

Purpose: A central file to store all project configurations, such as file paths, model hyerparameters and constants
"""

import os

portfolio = {
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CMCSA', 'NFLX', 'XOM'],
    'start_date': '2010-01-01',
    'end_date': '2023-01-01'
}

# Use absolute paths to ensure correct location from any working directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_paths = {
    'raw_data': os.path.join(project_root, 'data', 'raw'),
    'processed_data': os.path.join(project_root, 'data', 'processed'),
    'models': os.path.join(project_root, 'models'),
    'results': os.path.join(project_root, 'results')
}

results_path ={
    'figures': os.path.join(project_root, 'results', 'figures'),
    'tables': os.path.join(project_root, 'results', 'tables')
}
model_params = {
    'arima_order': (5, 1, 0),
    'garch_order': (1, 1),
    'lstm_units': 50,
    'lstm_epochs': 20,
    'lstm_batch_size': 32,
    'xgboost_params': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    },
    'catboost_params': {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 5
    },
    'cnn_params': {
        'filters': 64,
        'kernel_size': 3,
        'epochs': 20,
        'batch_size': 32
    }
}
