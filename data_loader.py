import pandas as pd
from pybaseball import statcast
from datetime import date
import os
from typing import Optional

class PitchDataLoader:
    """Class for loading and saving baseball pitch data."""
    
    def __init__(self, path: str, start_date: str = '2016-01-01', end_date: str = '2100-01-01') -> None:
        """
        Initializes the PitchDataLoader with the specified path and date range.

        Parameters:
        - path (str): The directory path where the pitch data is or will be stored.
        - start_date (str): The start date for the data range. Default is '2016-01-01'.
        - end_date (str): The end date for the data range. Default is '2100-01-01'.
        """
        self.path = path
        
        # Check if the parquet file exists and load it, else initialize an empty dataframe
        parquet_path = f'{self.path}pitch_data.parquet'
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True) if os.path.exists(parquet_path) else pd.DataFrame()
        
        self.start_date = pd.to_datetime(start_date).date()
        self.end_date = pd.to_datetime(end_date).date()
        
    def load_new_data(self) -> None:
        """
        Loads new pitch data from the specified date range and merges it with the existing data.
        """
        max_date = pd.to_datetime(self.df.game_date).max() if len(self.df) > 0 else self.start_date
        new_data = statcast(
            start_dt=max_date.strftime('%Y-%m-%d'), 
            end_dt=min(date.today(), self.end_date).strftime('%Y-%m-%d')
        ).reset_index(drop=True)
        
        if len(self.df) == 0:
            self.df = new_data
        elif len(new_data) > 0:
            self.df = pd.concat([self.df.loc[self.df.game_date < max_date], new_data]).reset_index(drop=True)
        
        self.save_parquet()
        
    def save_parquet(self, df_to_convert: Optional[pd.DataFrame] = None) -> None:
        """
        Saves the specified dataframe to a parquet file.

        Parameters:
        - df_to_convert (pd.DataFrame, optional): The dataframe to save. If not provided, uses the main dataframe.
        """
        if df_to_convert is None:
            df_to_convert = self.df
        
        df_to_convert.to_parquet(f'{self.path}pitch_data.parquet')