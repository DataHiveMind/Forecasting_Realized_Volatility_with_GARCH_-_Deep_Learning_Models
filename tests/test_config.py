import sys
sys.path.append('src')
from src.config import portfolio, data_paths, model_params

if __name__ == "__main__":
    # Test accessing portfolio configurations
    print("Portfolio configurations:")
    print(f"Tickers: {portfolio['tickers']}")
    print(f"Start Date: {portfolio['start_date']}")
    print(f"End Date: {portfolio['end_date']}")

    # Test accessing data_paths configurations
    print("\nData Paths configurations:")
    print(f"Raw Data Path: {data_paths['raw_data']}")
    print(f"Processed Data Path: {data_paths['processed_data']}")
    print(f"Models Path: {data_paths['models']}")
    print(f"Results Path: {data_paths['results']}")

    # Test accessing model_params configurations
    print("\nModel Parameters configurations:")
    print(f"ARIMA Order: {model_params['arima_order']}")
    print(f"GARCH Order: {model_params['garch_order']}")
    print(f"LSTM Units: {model_params['lstm_units']}")
    print(f"LSTM Epochs: {model_params['lstm_epochs']}")
    print(f"LSTM Batch Size: {model_params['lstm_batch_size']}")
    print(f"XGBoost Params: {model_params['xgboost_params']}")
    print(f"CatBoost Params: {model_params['catboost_params']}")
    print(f"CNN Params: {model_params['cnn_params']}")

    print("\nAll configurations imported and accessed successfully!")
