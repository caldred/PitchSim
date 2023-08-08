import pandas as pd

from data_loader import PitchDataLoader
import processing as prc
import stuff_model as stuff

path = 'data/'

pdl = PitchDataLoader(path=path)
pdl.load_new_data()
df = pdl.df
df = prc.apply_all_filters(df, prc.get_current_filters())
df = prc.save_memory(df, cols_to_drop=['des'])
df = prc.calculate_new_features(df)
df = prc.save_memory(df)
df.to_parquet(f'{path}mem_eff_pitch_data.parquet')

stuff_features = ['speed', 'speed_diff', 'lift', 'lift_diff', 'transverse_pit', 'transverse_pit_diff', 
                  'release_pos_x_pit', 'release_pos_y', 'release_pos_z', 'vert_approach_angle_adj']
cluster_target = 'csw'

df = stuff.assign_fuzzy_clusters(df, stuff_features, cluster_target)
df = prc.save_memory(df)
df.to_parquet(f'{path}clustered_pitch_data.parquet')

cluster_dist = stuff.create_location_distributions(df)
count_frequencies = stuff.calculate_count_frequencies(df)
platoon_cluster_dist = stuff.combine_flatten_distributions(cluster_dist, count_frequencies)
joblib.dump(platoon_cluster_dist, 'data/platoon_cluster_dist.dat')

feat = ['speed', 'vert_approach_angle_adj', 'transverse', 'transverse_pit', 'transverse_bat', 
        'lift', 'release_pos_x', 'release_pos_x_pit','release_pos_x_bat', 'release_pos_y', 
        'release_pos_z', 'plate_x', 'plate_x_pit','plate_x_bat', 'plate_x_abs', 'plate_z', 
        'plate_z_top', 'plate_z_bot', 'plate_dist', 'balls', 'strikes', 'speed_diff', 
        'lift_diff', 'transverse_pit_diff', 'vert_approach_angle', 'game_year']

model_filters = stuff.get_model_filters(df)
xgb_models = stuff.train_model(df, model_filters, feat)
joblib.dump(xgb_models, f'{path}xgb_models.dat')

batch_size = 10000
n_batches = 3
rv = pd.read_csv('runvalue.csv')

stuff.simulate_pitches(df, batch_size, n_batches, platoon_cluster_dist, 
                       feat, rv, xgb_models, path, year=2022)