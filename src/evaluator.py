"""
Time Series Model Evaluation

Comprehensive evaluation metrics and visualization for forecast models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class ForecastEvaluator:
    """
    Evaluate time series forecast models with multiple metrics.
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate comprehensive forecast accuracy metrics.
        
        Parameters
        ----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        
        Returns
        -------
        metrics : dict
            Dictionary of evaluation metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # sMAPE (symmetric MAPE)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'sMAPE': smape,
            'R2': r2
        }
    
    @staticmethod
    def plot_forecast(dates_train, y_train, dates_test, y_test, 
                     y_pred, y_lower=None, y_upper=None, title='Forecast Results'):
        """
        Plot actual vs predicted values with confidence intervals.
        
        Parameters
        ----------
        dates_train : array-like
            Training dates
        y_train : array-like
            Training values
        dates_test : array-like
            Test dates
        y_test : array-like
            Test actual values
        y_pred : array-like
            Test predictions
        y_lower : array-like, optional
            Lower confidence bound
        y_upper : array-like, optional
            Upper confidence bound
        title : str
            Plot title
        """
        plt.figure(figsize=(14, 6))
        
        # Plot training data
        plt.plot(dates_train, y_train, label='Training Data', color='blue', alpha=0.7)
        
        # Plot test data
        plt.plot(dates_test, y_test, label='Actual', color='green', marker='o', linewidth=2)
        
        # Plot predictions
        plt.plot(dates_test, y_pred, label='Forecast', color='red', marker='x', linewidth=2)
        
        # Plot confidence intervals
        if y_lower is not None and y_upper is not None:
            plt.fill_between(dates_test, y_lower, y_upper, color='red', alpha=0.2, label='95% CI')
        
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_residuals(residuals, title='Residual Analysis'):
        """
        Plot residual diagnostics.
        
        Parameters
        ----------
        residuals : array-like
            Model residuals
        title : str
            Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residual plot
        axes[0, 0].plot(residuals)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF plot
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=40, ax=axes[1, 1])
        axes[1, 1].set_title('Autocorrelation of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=1.00)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def rolling_window_validation(forecaster, dates, values, 
                                  window_size=120, horizon=12, step=12):
        """
        Perform rolling window cross-validation.
        
        Parameters
        ----------
        forecaster : object
            Forecaster instance with fit() and predict() methods
        dates : array-like
            Full date range
        values : array-like
            Full time series values
        window_size : int
            Size of training window
        horizon : int
            Forecast horizon
        step : int
            Step size between windows
        
        Returns
        -------
        results : DataFrame
            Cross-validation results with metrics for each fold
        """
        results = []
        
        n = len(values)
        for i in range(window_size, n - horizon, step):
            # Split data
            train_dates = dates[i-window_size:i]
            train_values = values[i-window_size:i]
            test_dates = dates[i:i+horizon]
            test_values = values[i:i+horizon]
            
            # Fit and predict
            try:
                forecaster.fit(train_dates, train_values)
                pred = forecaster.predict(steps=horizon)
                
                if isinstance(pred, pd.DataFrame):
                    pred_values = pred['forecast'].values
                else:
                    pred_values = pred.values if isinstance(pred, pd.Series) else pred
                
                # Calculate metrics
                metrics = ForecastEvaluator.calculate_metrics(test_values, pred_values)
                metrics['fold'] = len(results) + 1
                metrics['train_end'] = train_dates[-1]
                metrics['test_start'] = test_dates[0]
                
                results.append(metrics)
            except:
                continue
        
        return pd.DataFrame(results)
