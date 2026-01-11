"""
Facebook Prophet Forecasting Implementation

Wrapper for Prophet with business-friendly defaults and custom seasonality.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')


class ProphetForecaster:
    """
    Facebook Prophet forecaster with business time series optimizations.
    
    Parameters
    ----------
    seasonality_mode : str, default='additive'
        'additive' or 'multiplicative' seasonality
    changepoint_prior_scale : float, default=0.05
        Flexibility of trend changes (higher = more flexible)
    seasonality_prior_scale : float, default=10.0
        Strength of seasonality model
    yearly_seasonality : bool or int, default='auto'
        Fit yearly seasonality
    weekly_seasonality : bool or int, default='auto'
        Fit weekly seasonality
    daily_seasonality : bool or int, default='auto'
        Fit daily seasonality
    """
    
    def __init__(self, 
                 seasonality_mode='additive',
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
                 yearly_seasonality='auto',
                 weekly_seasonality='auto',
                 daily_seasonality='auto'):
        
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        
    def fit(self, dates, values, holidays=None):
        """
        Fit Prophet model to time series data.
        
        Parameters
        ----------
        dates : array-like
            Date/time index
        values : array-like
            Time series values
        holidays : DataFrame, optional
            Holiday definitions with columns 'holiday' and 'ds'
        """
        # Prepare data in Prophet format
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': values
        })
        
        # Initialize model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=holidays
        )
        
        # Fit model
        self.model.fit(df)
        
        return self
    
    def add_seasonality(self, name, period, fourier_order):
        """
        Add custom seasonality component.
        
        Parameters
        ----------
        name : str
            Name of seasonality component
        period : float
            Period in days
        fourier_order : int
            Number of Fourier terms
        """
        if self.model is None:
            raise ValueError("Must create model before adding seasonality")
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order
        )
    
    def predict(self, steps=12, freq='D'):
        """
        Generate forecasts for future time steps.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        freq : str
            Frequency of predictions ('D', 'W', 'M', 'Y', etc.)
        
        Returns
        -------
        forecast : DataFrame
            Forecasts with trend, seasonality components and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Return only future predictions
        return forecast.tail(steps)[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 
                                     'trend', 'yearly', 'weekly']]
    
    def cross_validate_model(self, initial='730 days', period='180 days', 
                            horizon='90 days'):
        """
        Perform time series cross-validation.
        
        Parameters
        ----------
        initial : str
            Initial training period
        period : str
            Period between cutoff dates
        horizon : str
            Forecast horizon
        
        Returns
        -------
        metrics : DataFrame
            Performance metrics (RMSE, MAE, MAPE)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before cross-validation")
        
        # Perform cross-validation
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        # Calculate metrics
        metrics = performance_metrics(df_cv)
        
        return metrics
    
    def get_changepoints(self):
        """
        Get detected trend changepoints.
        
        Returns
        -------
        changepoints : DataFrame
            Changepoint dates and magnitudes
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame({
            'changepoint': self.model.changepoints,
            'delta': self.model.params['delta'][0]
        })
    
    def decompose(self, dates, values):
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Returns
        -------
        components : DataFrame
            Decomposed components
        """
        if self.model is None:
            # Fit model if not already fitted
            self.fit(dates, values)
        
        # Get full prediction on training data
        df = pd.DataFrame({
            'ds': pd.to_datetime(dates),
            'y': values
        })
        
        forecast = self.model.predict(df)
        
        # Extract components
        components = forecast[['ds', 'trend', 'yearly', 'weekly', 'yhat']].copy()
        components['residual'] = values - forecast['yhat'].values
        
        return components
