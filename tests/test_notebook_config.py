import os
import sys

# Simulate the notebook's working directory (notebooks/)
os.chdir('notebooks')

# Add the parent directory to the Python path to access src
sys.path.append('..')

# Import configurations from src.config
from src.config import portfolio, data_paths, model_params

# Test accessing all configuration sections
print("Portfolio configurations:")
print(f"Tickers: {portfolio['tickers']}")
print(f"Start Date: {portfolio['start_date']}")
print(f"End Date: {portfolio['end_date']}")

print("\nData Paths configurations:")
print(f"Raw Data Path: {data_paths['raw_data']}")
print(f"Processed Data Path: {data_paths['processed_data']}")
print(f"Models Path: {data_paths['models']}")
print(f"Results Path: {data_paths['results']}")

print("\nModel Parameters configurations:")
print(f"ARIMA Order: {model_params['arima_order']}")
print(f"GARCH Order: {model_params['garch_order']}")
print(f"LSTM Units: {model_params['lstm_units']}")
print(f"LSTM Epochs: {model_params['lstm_epochs']}")
print(f"LSTM Batch Size: {model_params['lstm_batch_size']}")
print(f"XGBoost Params: {model_params['xgboost_params']}")
print(f"CatBoost Params: {model_params['catboost_params']}")
print(f"CNN Params: {model_params['cnn_params']}")

print("\nAll configurations imported and accessed successfully in simulated notebook environment!")
