"""
src/models.py

Purpose: Defines data models and structures used in the application such as GARCH models and ARIMA models, using tensorflow and keras for LSTM models.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import logging
import xgboost as xgb
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO)

def fit_arima_model(data: pd.Series, order: Tuple[int, int, int]) -> ARIMA:
    """
    Fit an ARIMA model to the provided time series data.

    Parameters:
    data (pd.Series): Time series data.
    order (Tuple[int, int, int]): The (p, d, q) order of the ARIMA model.

    Returns:
    ARIMA: Fitted ARIMA model.
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    logging.info(f"ARIMA model fitted with order {order}")
    return fitted_model

def fit_garch_model(data: pd.Series, p: int, q: int) -> arch_model:
    """
    Fit a GARCH model to the provided time series data.

    Parameters:
    data (pd.Series): Time series data.
    p (int): The order of the GARCH terms.
    q (int): The order of the ARCH terms.

    Returns:
    arch_model: Fitted GARCH model.
    """
    model = arch_model(data, vol='Garch', p=p, q=q)
    fitted_model = model.fit(disp='off')
    logging.info(f"GARCH model fitted with p={p}, q={q}")
    return fitted_model

def create_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """
    Create and compile an LSTM model.

    Parameters:
    input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).

    Returns:
    Sequential: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info(f"LSTM model created with input shape {input_shape}")
    return model

def prepare_lstm_data(data: pd.Series, time_step: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for LSTM model training.

    Parameters:
    data (pd.Series): Time series data.
    time_step (int): Number of time steps to look back.

    Returns:
    Tuple[np.ndarray, np.ndarray, MinMaxScaler]: Prepared input and output data along with the scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    logging.info(f"LSTM data prepared with time step {time_step}")
    return X, y, scaler

def compute_ewma_volatility(data: pd.Series, span: int) -> pd.Series:
    """
    Compute Exponential Weighted Moving Average volatility.

    Parameters:
    data (pd.Series): Time series data (returns).
    span (int): Span for EWMA.

    Returns:
    pd.Series: EWMA volatility.
    """
    ewma_vol = data.ewm(span=span).std()
    logging.info(f"EWMA volatility computed with span {span}")
    return ewma_vol

def fit_xgboost_model(X_train: np.ndarray, y_train: np.ndarray, params: dict ) -> xgb.XGBRegressor:
    """
    Fit an XGBoost model.

    Parameters:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    params (dict): Model parameters.

    Returns:
    xgb.XGBRegressor: Fitted XGBoost model.
    """
    if params is None:
        params = {'objective': 'reg:squarederror', 'n_estimators': 100}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    logging.info("XGBoost model fitted")
    return model

def fit_catboost_model(X_train: np.ndarray, y_train: np.ndarray, params: dict ) -> CatBoostRegressor:
    """
    Fit a CatBoost model.

    Parameters:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training targets.
    params (dict): Model parameters.

    Returns:
    CatBoostRegressor: Fitted CatBoost model.
    """
    if params is None:
        params = {'iterations': 100, 'verbose': 0}
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    logging.info("CatBoost model fitted")
    return model

def create_cnn_model(input_shape: Tuple[int, int]) -> Sequential:
    """
    Create and compile a 1D CNN model.

    Parameters:
    input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).

    Returns:
    Sequential: Compiled CNN model.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info(f"CNN model created with input shape {input_shape}")
    return model

def prepare_cnn_data(data: pd.Series, time_step: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare data for CNN model training.

    Parameters:
    data (pd.Series): Time series data.
    time_step (int): Number of time steps to look back.

    Returns:
    Tuple[np.ndarray, np.ndarray, MinMaxScaler]: Prepared input and output data along with the scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    logging.info(f"CNN data prepared with time step {time_step}")
    return X, y, scaler

if __name__ == "__main__":
    # Example usage
    from data_processing import load_raw_data, clean_data, save_data

    # Fetch data
    ticker = "AAPL"
    data = load_raw_data(ticker, "2020-01-01", "2021-01-01")
    data = clean_data(data)
    save_data(data, "data/processed/example.csv")
    data = data['Close']

    # Fit ARIMA model
    arima_order = (5, 1, 0)
    arima_model = fit_arima_model(data, arima_order)
    print(arima_model.summary())

    # Fit GARCH model
    garch_model = fit_garch_model(data.pct_change().dropna(), p=1, q=1)
    print(garch_model.summary())

    # Compute EWMA volatility
    returns = data.pct_change().dropna()
    ewma_vol = compute_ewma_volatility(returns, span=30)
    print(ewma_vol.tail())

    # Prepare LSTM data
    time_step = 10
    X, y, scaler = prepare_lstm_data(data, time_step)

    # Create and train LSTM model
    lstm_model = create_lstm_model((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=10, batch_size=32)
    logging.info("LSTM model training completed")

    # Fit XGBoost model
    xgboost_model = fit_xgboost_model(X, y, params={})
    print("XGBoost model fitted")

    # Fit CatBoost model
    catboost_model = fit_catboost_model(X, y, params={})
    print("CatBoost model fitted")

    # Prepare CNN data
    X_cnn, y_cnn, scaler_cnn = prepare_cnn_data(data, time_step)

    # Create and train CNN model
    cnn_model = create_cnn_model((X_cnn.shape[1], X_cnn.shape[2]))
    cnn_model.fit(X_cnn, y_cnn, epochs=10, batch_size=32)
    logging.info("CNN model training completed")
