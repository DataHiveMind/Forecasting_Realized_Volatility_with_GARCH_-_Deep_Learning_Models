# Forecasting Realized Volatility with GARCH & Deep Learning Models

This project explores forecasting realized volatility of financial assets using a combination of traditional econometric models (GARCH, ARIMA) and modern machine learning techniques (XGBoost, CatBoost, LSTM, CNN). The goal is to compare model performance and draw insights for volatility prediction.

## Project Structure

- **data/**: Contains raw and processed financial data for multiple tickers.
- **notebooks/**: Jupyter notebooks for each stage of the project:
  - `01_data_exploration.ipynb`: Data loading, quality checks, and visualization.
  - `02_feature_engineering.ipynb`: Feature creation including realized volatility, lagged features, and rolling statistics.
  - `03_model_prototyping.ipynb`: Model development, sanity checks, hyperparameter tuning, and rapid iteration.
  - `04_results_visualization.ipynb`: Loading predictions, comparison plots, scatter plots, performance metrics, and conclusions.
- **src/**: Source code for data processing, model definitions, training, and evaluation.
- **results/**: Output results including figures, tables, and summary conclusions.
- **models/**: Saved model files.
- **reports/**: Additional reports and documentation.

## Key Results (from Results Visualization)

### Model Performance Metrics

| Model   | MSE      | MAE    | RMSE   |
| ------- | -------- | ------ | ------ |
| ARIMA   | 9.30e-05 | 0.0077 | 0.0096 |
| GARCH   | 9.59e-05 | 0.0078 | 0.0098 |
| XGBoost | 9.93e-05 | 0.0079 | 0.0099 |

- **Best Model**: ARIMA (RMSE: 0.0096)
- Models evaluated: ARIMA, GARCH, XGBoost
- Visualizations include time series comparison plots and scatter plots of predicted vs actual realized volatility
- Key Insights:
  - ARIMA performed best among the evaluated models
  - XGBoost shows promise for capturing complex patterns
  - Further improvements possible with hyperparameter tuning and additional features

## Usage

1. Run the notebooks sequentially to reproduce the analysis.
2. Modify parameters in `src/config.py` to customize data paths and model settings.
3. Use the `src/models.py` module for model training and evaluation functions.
4. Results and figures are saved in the `results/` directory.

## Dependencies

- Python 3.x
- pandas, numpy, matplotlib
- statsmodels, arch
- scikit-learn
- xgboost
- catboost
- tensorflow (for LSTM and CNN models)

Install dependencies via:

```
pip install -r requirements.txt
```

## Conclusion

This project demonstrates a comprehensive approach to forecasting realized volatility using both classical and modern methods. The results show that ARIMA performed best among the evaluated models, though XGBoost shows promise for capturing complex patterns. Future work could explore hyperparameter tuning, additional features, deep learning architectures, and ensemble techniques for further improvements.
