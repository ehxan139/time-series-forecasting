"""
Utility functions for time series preprocessing and analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats


def detect_outliers(series, method='iqr', threshold=1.5):
    """
    Detect outliers in time series.

    Parameters
    ----------
    series : array-like
        Time series values
    method : str
        Detection method: 'iqr', 'zscore', or 'modified_zscore'
    threshold : float
        Threshold for outlier detection

    Returns
    -------
    outlier_indices : array
        Boolean array indicating outlier positions
    """
    series = np.array(series)

    if method == 'iqr':
        Q1 = np.percentile(series, 25)
        Q3 = np.percentile(series, 75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = (series < lower) | (series > upper)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series))
        outliers = z_scores > threshold

    elif method == 'modified_zscore':
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        outliers = np.abs(modified_z_scores) > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    return outliers


def impute_missing(series, method='linear'):
    """
    Impute missing values in time series.

    Parameters
    ----------
    series : pd.Series
        Time series with missing values
    method : str
        Imputation method: 'linear', 'forward', 'backward', 'mean', 'median'

    Returns
    -------
    imputed : pd.Series
        Series with imputed values
    """
    if method == 'linear':
        return series.interpolate(method='linear')
    elif method == 'forward':
        return series.fillna(method='ffill')
    elif method == 'backward':
        return series.fillna(method='bfill')
    elif method == 'mean':
        return series.fillna(series.mean())
    elif method == 'median':
        return series.fillna(series.median())
    else:
        raise ValueError(f"Unknown method: {method}")


def decompose_series(series, period=12, model='additive'):
    """
    Decompose time series into trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
        Time series to decompose
    period : int
        Seasonal period
    model : str
        'additive' or 'multiplicative'

    Returns
    -------
    components : dict
        Dictionary with trend, seasonal, and residual components
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')

    return {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'residual': result.resid,
        'observed': result.observed
    }


def create_features(dates, values):
    """
    Create time-based features for time series.

    Parameters
    ----------
    dates : array-like
        Date/time index
    values : array-like
        Time series values

    Returns
    -------
    features : pd.DataFrame
        DataFrame with engineered features
    """
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'value': values
    })

    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Lag features
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Rolling statistics
    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()

    return df
