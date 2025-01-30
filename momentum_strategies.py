import datetime
import numpy as np
import seaborn as sns
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import datetime



def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    returns = srs / srs.shift(day_offset) - 1.0
    return returns


class TSMOM_strategy:
    def __init__(self, srs, VOL_LOOKBACK, VOL_TARGET, volatility_scaling=True):
        self.srs = srs
        self.VOL_LOOKBACK = VOL_LOOKBACK
        self.VOL_TARGET = VOL_TARGET
        self.volatility_scaling = volatility_scaling

    def trend_estimation(self, TS_LENGTH):
        """
        This function provides trend estimation for position sizing.
        """
        trend_estimation = calc_returns(self.srs, TS_LENGTH)
        return trend_estimation

    def position_sizing(self, trend, activation='sign'):
        """
        Returns position sizing values ranging from [-1, 1], indicating long, short, or do nothing.
        """
        if activation == 'sign':
            signal = np.maximum(0, np.sign(trend))
        elif activation == 'tanh':
            signal = np.maximum(0, np.tanh(trend))
        return signal

    def calc_daily_vol(self, daily_returns):
        return (
            daily_returns.ewm(span=self.VOL_LOOKBACK, min_periods=self.VOL_LOOKBACK)
            .std()
            .fillna(method="bfill")
        )

    def calc_vol_scaled_returns(self, daily_returns):
        daily_vol = self.calc_daily_vol(daily_returns)
        annualised_vol = daily_vol * np.sqrt(252)  # annualised
        position_map = self.VOL_TARGET / annualised_vol.shift(1)

        # Debugging
        print("Position Map Columns:", position_map.columns)
        print("Position Map Preview:\n", position_map.head())
        
        # Ensure proper column access and cap scaling factor
        if 'Close' not in position_map.columns:
            raise ValueError("The 'Close' column is missing in position_map. Please check the data.")

        position_map['Close'] = np.where(position_map['Close'] > 2.0, 2.0, position_map['Close'])

        return daily_returns * position_map
    
    def volatility_target_map(self):
        daily_vol = self.calc_daily_vol(calc_returns(self.srs))
        annualised_vol = daily_vol * np.sqrt(252)  # annualised
        position_map = self.VOL_TARGET / annualised_vol.shift(1)

        # Debugging
        print("Position Map Columns:", position_map.columns)
        print("Position Map Preview:\n", position_map.head())

        # Ensure proper column access and cap scaling factor
        if 'Close' not in position_map.columns:
            raise ValueError("The 'Close' column is missing in position_map. Please check the data.")

        position_map['Close'] = np.where(position_map['Close'] > 2.0, 2.0, position_map['Close'])

        return position_map

    def cal_strategy_returns(self, signal):
        """
        Calculates returns based on position sizing.
        """
        daily_returns = calc_returns(self.srs)
        next_day_returns = (
            self.calc_vol_scaled_returns(daily_returns).shift(-1)
            if self.volatility_scaling
            else daily_returns.shift(-1)
        )
        cap_returns = signal * next_day_returns
        return cap_returns
    
    