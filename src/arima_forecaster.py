"""
ARIMA/SARIMA Forecasting Implementation

Provides automatic ARIMA model selection and forecasting with seasonal support.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import warnings
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA/SARIMA forecaster with automatic order selection.

    Parameters
    ----------
    seasonal : bool, default=False
        Whether to use seasonal ARIMA (SARIMA)
    m : int, default=12
        Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    auto_select : bool, default=True
        Automatically select best ARIMA orders using AIC
    max_p : int, default=5
        Maximum AR order to try
    max_d : int, default=2
        Maximum differencing order to try
    max_q : int, default=5
        Maximum MA order to try
    """

    def __init__(self, seasonal=False, m=12, auto_select=True,
                 max_p=5, max_d=2, max_q=5):
        self.seasonal = seasonal
        self.m = m
        self.auto_select = auto_select
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.model_fit = None
        self.best_order = None
        self.best_seasonal_order = None

    def check_stationarity(self, series):
        """
        Check if time series is stationary using ADF and KPSS tests.

        Returns
        -------
        dict with test results and recommendations
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        adf_stationary = adf_result[1] < 0.05

        # KPSS test
        kpss_result = kpss(series.dropna(), regression='ct')
        kpss_stationary = kpss_result[1] > 0.05

        return {
            'adf_pvalue': adf_result[1],
            'adf_stationary': adf_stationary,
            'kpss_pvalue': kpss_result[1],
            'kpss_stationary': kpss_stationary,
            'is_stationary': adf_stationary and kpss_stationary,
            'differencing_needed': not (adf_stationary and kpss_stationary)
        }

    def _find_best_order(self, series):
        """
        Find best ARIMA order using grid search and AIC.
        """
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None

        # Define order ranges
        p_range = range(0, self.max_p + 1)
        d_range = range(0, self.max_d + 1)
        q_range = range(0, self.max_q + 1)

        # Test combinations
        for p, d, q in itertools.product(p_range, d_range, q_range):
            try:
                if self.seasonal:
                    # Try seasonal orders (P, D, Q)
                    for P, D, Q in itertools.product([0, 1], [0, 1], [0, 1]):
                        if P == 0 and D == 0 and Q == 0:
                            seasonal_order = (0, 0, 0, 0)
                        else:
                            seasonal_order = (P, D, Q, self.m)

                        model = SARIMAX(
                            series,
                            order=(p, d, q),
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted = model.fit(disp=False, maxiter=200)

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            best_seasonal_order = seasonal_order
                else:
                    model = SARIMAX(
                        series,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted = model.fit(disp=False, maxiter=200)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)

            except:
                continue

        return best_order, best_seasonal_order, best_aic

    def fit(self, dates, values, order=None, seasonal_order=None):
        """
        Fit ARIMA model to time series data.

        Parameters
        ----------
        dates : array-like
            Date/time index
        values : array-like
            Time series values
        order : tuple, optional
            ARIMA order (p, d, q). If None, automatically selected.
        seasonal_order : tuple, optional
            Seasonal order (P, D, Q, m). If None, automatically selected.
        """
        # Create time series
        if not isinstance(values, pd.Series):
            ts = pd.Series(values, index=pd.to_datetime(dates))
        else:
            ts = values

        # Auto-select order if not provided
        if self.auto_select and order is None:
            print("Selecting best ARIMA order...")
            self.best_order, self.best_seasonal_order, aic = self._find_best_order(ts)
            print(f"Best order: {self.best_order}, AIC: {aic:.2f}")
            if self.seasonal:
                print(f"Best seasonal order: {self.best_seasonal_order}")
        else:
            self.best_order = order or (1, 1, 1)
            self.best_seasonal_order = seasonal_order or (0, 0, 0, 0)

        # Fit model
        if self.seasonal:
            self.model = SARIMAX(
                ts,
                order=self.best_order,
                seasonal_order=self.best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            self.model = SARIMAX(
                ts,
                order=self.best_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

        self.model_fit = self.model.fit(disp=False, maxiter=200)

        return self

    def predict(self, steps=12, return_conf_int=True, alpha=0.05):
        """
        Generate forecasts for future time steps.

        Parameters
        ----------
        steps : int
            Number of steps to forecast
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals

        Returns
        -------
        forecast : array or DataFrame
            Point forecasts and confidence intervals if requested
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before prediction")

        # Generate forecast
        forecast = self.model_fit.forecast(steps=steps)

        if return_conf_int:
            # Get prediction intervals
            pred = self.model_fit.get_prediction(
                start=len(self.model_fit.fittedvalues),
                end=len(self.model_fit.fittedvalues) + steps - 1
            )
            pred_df = pred.summary_frame(alpha=alpha)

            return pd.DataFrame({
                'forecast': forecast.values,
                'lower': pred_df['mean_ci_lower'].values,
                'upper': pred_df['mean_ci_upper'].values
            })

        return forecast

    def get_residuals(self):
        """Get model residuals for diagnostics."""
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
        return self.model_fit.resid

    def summary(self):
        """Print model summary."""
        if self.model_fit is None:
            raise ValueError("Model must be fitted first")
        return self.model_fit.summary()
