"""
Exponential Smoothing Methods

Fast baseline forecasting using Simple, Holt, and Holt-Winters exponential smoothing.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt


class ExponentialSmoothingForecaster:
    """
    Exponential smoothing forecaster with multiple methods.

    Parameters
    ----------
    method : str, default='holt_winters'
        Smoothing method: 'simple', 'holt', or 'holt_winters'
    seasonal : str, optional
        Seasonal component type: 'add' or 'mul'
    seasonal_periods : int, optional
        Number of periods in seasonal cycle
    trend : str, optional
        Trend component type: 'add' or 'mul'
    """

    def __init__(self, method='holt_winters', seasonal=None,
                 seasonal_periods=None, trend='add'):
        self.method = method.lower()
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.model = None
        self.model_fit = None

        # Validate method
        valid_methods = ['simple', 'holt', 'holt_winters']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def fit(self, dates, values):
        """
        Fit exponential smoothing model.

        Parameters
        ----------
        dates : array-like
            Date/time index
        values : array-like
            Time series values
        """
        # Create time series
        if not isinstance(values, pd.Series):
            ts = pd.Series(values, index=pd.to_datetime(dates))
        else:
            ts = values

        # Fit appropriate model
        if self.method == 'simple':
            self.model = SimpleExpSmoothing(ts)
            self.model_fit = self.model.fit()

        elif self.method == 'holt':
            self.model = Holt(ts)
            self.model_fit = self.model.fit()

        elif self.method == 'holt_winters':
            if self.seasonal is None or self.seasonal_periods is None:
                raise ValueError("Holt-Winters requires seasonal and seasonal_periods")

            self.model = ExponentialSmoothing(
                ts,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods
            )
            self.model_fit = self.model.fit()

        return self

    def predict(self, steps=12):
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int
            Number of steps to forecast

        Returns
        -------
        forecast : array
            Point forecasts
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def get_level_trend_seasonal(self):
        """
        Get smoothed level, trend, and seasonal components.

        Returns
        -------
        components : dict
            Dictionary with level, trend, and seasonal components
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")

        components = {'level': self.model_fit.level}

        if self.method in ['holt', 'holt_winters']:
            components['trend'] = self.model_fit.slope

        if self.method == 'holt_winters':
            components['seasonal'] = self.model_fit.season

        return components
