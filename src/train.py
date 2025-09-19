"""
src/train.py 

Purpose: This module imports data and models, then trains various models including GARCH, LSTM, XGBoost, CatBoost, and CNN on the processed data.
"""
import sys
import os
import pandas as pd
from src.config import portfolio, data_paths, model_params
from src.data_processing import clean_data, add_technical_indicators
from src.models import (
    fit_garch_model,
    create_lstm_model,
    prepare_lstm_data,
    fit_xgboost_model,
    fit_catboost_model,
    create_cnn_model,
    prepare_cnn_data
)

def load_processed_data(ticker):
    path = os.path.join(data_paths['processed_data'], f"{ticker}_processed.csv")
    return pd.read_csv(path)

def train_all_models():
    tickers = portfolio['tickers']
    for ticker in tickers:
        print(f"Training models for {ticker}...")
        df = load_processed_data(ticker)
        df = clean_data(df)
        df = add_technical_indicators(df, short=25, long=50)

        # GARCH
        returns = df['Close'].pct_change().dropna()
        garch_order = model_params['garch_order']
        garch_model = fit_garch_model(returns, p=garch_order[0], q=garch_order[1])

        # LSTM
        time_step = 10
        X_lstm, y_lstm, scaler_lstm = prepare_lstm_data(df['Close'], time_step)
        lstm_model = create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        lstm_model.fit(X_lstm, y_lstm, epochs=model_params['lstm_epochs'], batch_size=model_params['lstm_batch_size'], verbose=0)

        # Reshape for XGBoost and CatBoost (flatten last two dimensions)
        X_flat = X_lstm.reshape(X_lstm.shape[0], -1)

        # XGBoost
        xgb_params = model_params['xgboost_params']
        xgb_model = fit_xgboost_model(X_flat, y_lstm, xgb_params)

        # CatBoost
        cat_params = model_params['catboost_params']
        cat_model = fit_catboost_model(X_flat, y_lstm, cat_params)

        # CNN
        X_cnn, y_cnn, scaler_cnn = prepare_cnn_data(df['Close'], time_step)
        cnn_model = create_cnn_model((X_cnn.shape[1], X_cnn.shape[2]))
        cnn_model.fit(X_cnn, y_cnn, epochs=model_params['cnn_params']['epochs'], batch_size=model_params['cnn_params']['batch_size'], verbose=0)

        print(f"Finished training for {ticker}.")

if __name__ == "__main__":
    train_all_models()