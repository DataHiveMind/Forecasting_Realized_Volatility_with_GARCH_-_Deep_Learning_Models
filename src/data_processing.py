"""
src/data_processing.py

Purpose: This module contains functions for processing and cleaning data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from src.config import data_paths

def load_raw_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing historical stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    if data is None:
        return pd.DataFrame()
    raw_file_path = os.path.join(data_paths['raw_data'], f'raw_{ticker}_data.csv')
    os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
    data.to_csv(raw_file_path)
    return data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the stock data by handling missing values and removing duplicates, filling missing values, and handling outliers. and save the cleaned data to a CSV file with a specified naming convention.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicates and fill missing values
    df = df.drop_duplicates()
    df = df.ffill().bfill()

    # Handle outliers by capping values at 1.5*IQR
    numeric_df = df.select_dtypes(include=[np.number])
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    # Keep rows where all numeric columns are within IQR bounds
    mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
    cleaned_df = df[mask]
    cleaned_df = cleaned_df.dropna()

    # Save cleaned data with specified naming convention
    processed_file_path = os.path.join(data_paths['processed_data'], 'cleaned_data.csv')
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
    df.to_csv(processed_file_path, index=True)
    return df

def add_technical_indicators(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.

    Parameters:
    df (pd.DataFrame): DataFrame containing stock data.
    short (int): Short window for moving average.
    long (int): Long window for moving average.

    Returns:
    pd.DataFrame: DataFrame with added technical indicators.
    """
    if df.empty or 'Close' not in df.columns:
        return df

    # Ensure 'Close' is numeric and handle MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        close_col = ('Close', df.columns.get_level_values(1)[0]) if 'Close' in df.columns.get_level_values(0) else 'Close'
    else:
        close_col = 'Close'

    if close_col in df.columns:
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df[f'SMA_{short}'] = df[close_col].rolling(window=short).mean()
        df[f'SMA_{long}'] = df[close_col].rolling(window=long).mean()
        df[f'EMA_{short}'] = df[close_col].ewm(span=short, adjust=False).mean()
        df[f'EMA_{long}'] = df[close_col].ewm(span=long, adjust=False).mean()
        df = df.dropna()
    return df

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the processed DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): DataFrame to be saved.
    file_path (str): Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=True)

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2021-01-01"
    short_window = 20
    long_window = 50
    output_file = os.path.join(data_paths['processed_data'], 'processed_data.csv')

    data = load_raw_data(ticker, start_date, end_date)
    cleaned_data = clean_data(data)
    processed_data = add_technical_indicators(cleaned_data, short_window, long_window)
    save_data(processed_data, output_file)
