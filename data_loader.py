import pandas as pd
from pybaseball import statcast, schedule_and_record
from datetime import date
import os

class PitchDataLoader():
    
    def __init__(self, path, start_date='2016-01-01', end_date='2100-01-01'):
        self.path = path
        if os.path.exists(f'{self.path}pitch_data.parquet'):
            self.df = pd.read_parquet(f'{self.path}pitch_data.parquet').reset_index(drop=True)
        else:
            self.df = pd.DataFrame()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
    def load_new_data(self):
        if len(self.df) > 0:
            max_date = pd.to_datetime(self.df.game_date).max()
        else:
            max_date = self.start_date
        new_data = statcast(max_date.strftime('%Y-%m-%d'), 
                           min(date.today(), self.end_date).strftime('%Y-%m-%d')
                           ).reset_index(drop=True)
        if len(self.df) == 0:
            self.df = new_data
        elif len(new_data) > 0:
            self.df = pd.concat([self.df.loc[self.df.game_date < max_date], 
                                new_data]).reset_index(drop=True)
        self.save_parquet()
        
    def save_parquet(self, df_to_convert=None):
        if df_to_convert is None:
            df_to_convert = self.df
        df_to_convert.to_parquet(f'{self.path}pitch_data.parquet')

