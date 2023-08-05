from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype
from sklearn import linear_model

@dataclass
class Filter:
    col: str
    min: float = np.NINF
    max: float = np.PINF
    keep_na: bool = True
    include: str = None
    exclude: str = None

def apply_all_filters(df, filters):
    indices = []
    for filter in filters:
        if is_numeric_dtype(df[filter.col]):
            indices.append((df[filter.col] >= filter.min))
            indices.append((df[filter.col] <= filter.max))
        if not filter.keep_na:
            indices.append((~df[filter.col].isna()))
        if filter.include != None:
            indices.append((df[filter.col].isin(filter.include)))
        if filter.exclude != None:
            indices.append((~df[filter.col].isin(filter.exclude)))
    indices_array = np.array([x.values for x in indices])
    df = df.loc[np.logical_and.reduce(indices_array, axis=0)].reset_index(drop=True)
    return df

def default_filters():
    filters = []
    filters.append(Filter(col='release_speed', min=45, max=107, keep_na=False))
    filters.append(Filter(col='release_pos_x', min=-10, max=10, keep_na=False))
    filters.append(Filter(col='release_pos_z', min=0, max=10, keep_na=False))
    filters.append(Filter(col='pfx_x', min=-3, max=3, keep_na=False))
    filters.append(Filter(col='pfx_z', min=-2, max=3, keep_na=False))
    filters.append(Filter(col='plate_x', min=-8, max=8, keep_na=False))
    filters.append(Filter(col='plate_z', min=-4, max=8, keep_na=False))
    filters.append(Filter(col='vx0', min=-20, max=20, keep_na=False))
    filters.append(Filter(col='vy0', min=-160, max=-80, keep_na=False))
    filters.append(Filter(col='ax', min=-40, max=30, keep_na=False))
    filters.append(Filter(col='ay', min=5, max=45, keep_na=False))
    filters.append(Filter(col='az', min=-55, max=5, keep_na=False))
    filters.append(Filter(col='release_extension', min=3, max=9, keep_na=False))
    filters.append(Filter(col='balls', min=0, max=3, keep_na=False))
    filters.append(Filter(col='strikes', min=0, max=2, keep_na=False))
    filters.append(Filter(col='outs_when_up', min=0, max=2, keep_na=False))
    filters.append(Filter(col='inning', min=1, max=18, keep_na=False))
    filters.append(Filter(col='inning_topbot', include=['Bot', 'Top'], keep_na=False))
    filters.append(Filter(col='batter', keep_na=False))
    filters.append(Filter(col='pitcher', keep_na=False))
    filters.append(Filter(col='fielder_2', keep_na=False))
    return filters

def save_memory(df, float_overrides=None, category_overrides=None, cols_to_drop=None):
    df = df.drop_duplicates(subset=['game_pk', 'batter', 'pitcher', 'at_bat_number', 'pitch_number'])
    if float_overrides is None:
        float_overrides = [
            'spin_dir',
            'hit_distance_sc', 
            'launch_angle', 
            'release_spin_rate',
            'woba_denom', 
            'babip_value', 
            'iso_value', 
            'spin_axis'
        ]
    if category_overrides is None:
        category_overrides =  [
            'hit_location',
            'on_3b',
            'on_2b',
            'on_1b',
            'umpire',
            'launch_speed_angle'
        ]
    
    if cols_to_drop is not None:
        df = df.drop(columns = cols_to_drop)

    for col in float_overrides:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    for col in category_overrides:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    for col in df.columns:
        if df[col].isnull().all():
            df = df.drop(columns=[col])
            continue
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        if is_integer_dtype(df[col]):
            if np.nanmin(df[col]) >= 0:
                if np.nanmax(df[col]) < 2**8:
                    df[col] = df[col].astype('uint8')
                elif np.nanmax(df[col]) < 2**16:
                    df[col] = df[col].astype('uint16')
                elif np.nanmax(df[col]) < 2**32:
                    df[col] = df[col].astype('uint32')
                else:
                    df[col] = df[col].astype('uint64')
            else:
                if np.nanmax(df[col]) < 2**7:
                    df[col] = df[col].astype('int8')
                elif np.nanmax(df[col]) < 2**15:
                    df[col] = df[col].astype('int16')
                elif np.nanmax(df[col]) < 2**31:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    return df

def calculate_new_features(df):
    df['az_adj'] = df.az + 32.174
    df['speed'] = np.sqrt(np.square(df.vx0)+np.square(df.vy0)+np.square(df.vz0))
    df['drag'] = -(df.ax*df.vx0+df.ay*df.vy0+(df.az_adj)*df.vz0)/df.speed
    
    df['lift'] = np.cross(df[['ax', 'ay', 'az_adj']].values, df[['vx0', 'vy0', 'vz0']].values)[:,0]/df.speed
    
    df['transverse_x'] = np.cross(df[['ax', 'ay', 'az_adj']].values, 
                                df[['vx0', 'vy0', 'vz0']].values)[:,2]/df.speed
    df['transverse_y'] = np.cross(df[['ax', 'ay', 'az_adj']].values, 
                                df[['vx0', 'vy0', 'vz0']].values)[:,1]/df.speed
    df['transverse'] = np.sqrt(np.square(df.transverse_x) + np.square(df.transverse_y))*np.sign(df.transverse_x)

    mean_drag = np.mean(df.drag/np.square(df.speed))
    df['coeff_drag'] = df.drag/np.square(df.speed)/mean_drag*.35
    
    df['throws_r'] = np.where(df.p_throws == 'R', 1, 0)
    df['stand_r'] = np.where(df.stand == 'R', 1, 0)
    df['transverse_pit'] = df.transverse*(-2*df.throws_r+1)
    df['transverse_bat'] = df.transverse*(-2*df.stand_r+1)
    df['release_pos_x_pit'] = df.release_pos_x*(2*df.throws_r-1)
    df['release_pos_x_bat'] = df.release_pos_x*(2*df.stand_r-1)
    df['plate_x_pit']  = df.plate_x*(2*df.throws_r-1)
    df['plate_x_bat']  = df.plate_x*(2*df.stand_r-1)
    df['plate_x_abs'] = np.abs(df.plate_x)
    df['plate_z_top'] = df.plate_z-df.sz_top
    df['plate_z_bot'] = df.plate_z-df.sz_bot
    df['plate_dist'] = np.sqrt(np.square(df.plate_x) + np.square(df.plate_z-(df.sz_top+df.sz_bot)/2))
    df['release_pos_y'] = np.where(df.release_pos_y == 50, 60.5-df.release_extension, df.release_pos_y)

    df['pitch'] = 1
    df['swing'] = np.where(df.description.isin(['foul', 
                                                'hit_into_play', 
                                                'swinging_strike', 
                                                'hit_into_play_no_out', 
                                                'hit_into_play_score', 
                                                'foul_tip', 
                                                'swinging_strike_blocked']), 1, 0)
    df['swstr'] = np.where(df.description.isin(['swinging_strike', 
                                                'foul_tip', 
                                                'swinging_strike_blocked']), 1, 0)
    df['contact'] = df.swing-df.swstr
    df['foul'] = np.where(df.description == 'foul', 1, 0)
    df['inplay'] = df.contact-df.foul
    df['noswing'] = 1 - df.swing
    df['callstr'] = np.where(df.description == 'called_strike', 1, 0)
    df['ball'] = df.noswing-df.callstr

    df['game_date'] = pd.to_datetime(df.game_date)
    df['game_month'] = pd.Series(np.clip(pd.DatetimeIndex(df['game_date']).month,4,10))
    df['game_year'] = pd.Series(pd.DatetimeIndex(df['game_date']).year)
    
    df['is_on_3b'] = ~df.on_3b.isna()
    df['is_on_2b'] = ~df.on_2b.isna()
    df['is_on_1b'] = ~df.on_1b.isna()
    df['score_diff'] = (df.bat_score - df.fld_score).astype('int8')

    tto = df[['game_pk', 'batter', 'pitcher', 'at_bat_number']].drop_duplicates()
    tto = tto.sort_values(by=['game_pk', 'batter', 'pitcher', 'at_bat_number'])
    tto['tto'] = tto.groupby(['game_pk', 'batter', 'pitcher']).at_bat_number.cumcount()+1
    df = df.merge(tto, on=['game_pk', 'batter', 'pitcher', 'at_bat_number'], how='left')

    ps = df[['game_pk', 'batter', 'pitcher', 'at_bat_number', 'pitch_number']]
    ps = ps.sort_values(by=['game_pk', 'batter', 'pitcher', 'at_bat_number', 'pitch_number'])
    ps['pitches_seen'] = ps.groupby(['game_pk', 'batter', 'pitcher']).pitch_number.cumcount()+1
    ps = ps.sort_index()
    df['pitches_seen'] = ps.pitches_seen

    pc = df[['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']]
    pc = pc.sort_values(by=['game_pk', 'pitcher', 'at_bat_number', 'pitch_number'])
    pc['pitch_count'] = pc.groupby(['game_pk', 'pitcher']).pitch_number.cumcount()+1
    pc = pc.sort_index()
    df['pitch_count'] = pc.pitch_count

    tp = df[['pitcher', 'game_date', 'pitch_number']].groupby(['pitcher', 'game_date']).count().reset_index()
    tp.columns = ['pitcher', 'game_date', 'total_pitches']
    tps = {}
    for n in range(1,8):
        tps[n] = tp.copy()
        tps[n]['game_date'] += pd.Timedelta(days=n)
        df = df.merge(tps[n], on=['pitcher', 'game_date'], how='left', suffixes=('',f'_{n}'))
    df = df.rename(columns={'total_pitches':'total_pitches_1'})
    for n in range(1,8):
        df[f'total_pitches_{n}'] = df[f'total_pitches_{n}'].fillna(0).astype('uint8')

    df['inning_top'] = np.where(df.inning_topbot == 'Top', 1, 0)
    df['if_shift'] = np.where(df.if_fielding_alignment == 'Standard', 0,
                                np.where(df.if_fielding_alignment == 'Strategic', 1, 2))
    df['of_shift'] = np.where(df.of_fielding_alignment == 'Standard', 0,
                                np.where(df.of_fielding_alignment == 'Strategic', 1, 
                                        np.where(df.of_fielding_alignment == 'Extreme outfield shift', 2, 3)))

    cols = ['game_pk', 'batter', 'at_bat_number',
        'release_pos_x', 'release_pos_y', 'release_pos_z']
    grp_cols = ['game_pk', 'batter', 'at_bat_number']
    rp = df[cols].groupby(grp_cols).mean().reset_index()
    rp = rp.sort_values(by=['game_pk', 'batter', 'at_bat_number']).reset_index()
    rp1 = rp.copy()
    rp1['index'] += 1
    rp = rp.merge(rp1, on=['index', 'game_pk', 'batter'], how='left', suffixes=('', '_prior'))
    cols = ['game_pk', 'batter', 'at_bat_number', 'release_pos_x_prior',
            'release_pos_y_prior', 'release_pos_z_prior']
    df = df.merge(rp[cols], on=['game_pk', 'batter', 'at_bat_number'], how='left')
    df['release_pos_x_diff'] = df.release_pos_x - df.release_pos_x_prior
    df['release_pos_y_diff'] = df.release_pos_y - df.release_pos_y_prior
    df['release_pos_z_diff'] = df.release_pos_z - df.release_pos_z_prior
    df = df.drop(columns = ['release_pos_x_prior', 'release_pos_y_prior', 'release_pos_z_prior'])

    grp = df.loc[df.pitch_type.isin(['FF', 'SI', 'FT']) & (df.game_type == 'R'), 
                    ['speed', 'lift', 'transverse_pit', 'pitcher', 'game_year']
                ].groupby(by=['pitcher', 'game_year'])
    pfd = grp.mean().reset_index()
    pfd = df[['speed', 'lift', 'transverse_pit', 'pitcher', 'game_year']
            ].merge(pfd, how='left', on=['pitcher', 'game_year'], suffixes=('','_diff'))
    pfd['speed_diff'] = pfd.speed-pfd.speed_diff
    pfd['lift_diff'] = pfd.lift-pfd.lift_diff
    pfd['transverse_pit_diff'] = pfd.transverse_pit-pfd.transverse_pit_diff
    df['speed_diff'] = np.where(df.pitch_type.isin(['FF', 'SI', 'FT']), 0, 
                                np.clip(pfd.speed_diff, 0, -50))
    df['lift_diff'] = np.where(df.pitch_type.isin(['FF', 'SI', 'FT']), 0, 
                                np.clip(pfd.lift_diff, 0, -50))
    df['transverse_pit_diff'] = np.where(df.pitch_type.isin(['FF', 'SI', 'FT']), 
                                            0, pfd.transverse_pit_diff)

    cols = ['game_pk', 'at_bat_number', 'pitch_number', 'speed', 'transverse', 'lift', 'plate_x', 'plate_z']
    pp = df[cols]
    pp['pitch_number'] += 1
    df = df.merge(pp, on=['game_pk', 'at_bat_number', 'pitch_number'], how='left', suffixes=('', '_prior'))
    df['plate_x_prior_pit']  = df.plate_x_prior*(2*df.throws_r-1)
    df['plate_x_prior_bat']  = df.plate_x_prior*(2*df.stand_r-1)
    df['speed_prior_diff'] = df.speed - df.speed_prior
    df['transverse_prior_diff'] = df.transverse - df.transverse_prior
    df['lift_prior_diff'] = df.lift - df.lift_prior
    df['plate_x_prior_diff'] = df.plate_x - df.plate_x_prior
    df['plate_x_prior_pit_diff'] = df.plate_x_pit - df.plate_x_prior_pit
    df['plate_x_prior_bat_diff'] = df.plate_x_bat - df.plate_x_prior_bat
    df['plate_z_prior_diff'] = df.plate_z - df.plate_z_prior

    df['time_to_plate'] = abs((df.vy0 + np.sqrt(df.vy0**2 - 2 * df.ay * df.release_pos_y))/df.ay)
    df['speed_at_plate'] = np.sqrt((df.vx0 + df.ax*df.time_to_plate)**2 + (df.vy0 + df.ay*df.time_to_plate)**2 + (df.vz0 + df.az*df.time_to_plate)**2)
    df['vert_approach_angle'] = np.arctan((df.vz0 + df.az*df.time_to_plate)/(df.vy0 + df.ay*df.time_to_plate))
    df['horz_approach_angle'] = np.arctan((df.vx0 + df.ax*df.time_to_plate)/(df.vy0 + df.ay*df.time_to_plate))

    return df