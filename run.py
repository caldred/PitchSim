import pandas as pd
import joblib
from typing import List, Dict, Optional

from data_loader import PitchDataLoader
import processing as prc
import stuff_model as stuff

def pipeline(path: str, year: Optional[int] = None):
    """
    A pipeline to process pitch data, cluster the data, train models, make predictions, and generate results.
    
    Parameters:
    - path (str): Base path for saving and reading files.
    - year (Optional[int]): Year to filter the data on. If None, data isn't filtered on year.
    """
    
    # Data Loading and Processing
    pdl = PitchDataLoader(path=path)
    pdl.load_new_data()
    df = pdl.df
    df = prc.apply_all_filters(df, prc.get_current_filters())
    df = prc.save_memory(df, cols_to_drop=['des'])
    df = prc.calculate_new_features(df)
    df = prc.save_memory(df)
    df.to_parquet(f'{path}mem_eff_pitch_data.parquet')

    # Clustering
    stuff_features = ['speed', 'speed_diff', 'lift', 'lift_diff', 'transverse_pit', 
                      'transverse_pit_diff', 'release_pos_x_pit', 'release_pos_y', 
                      'release_pos_z', 'vert_approach_angle_adj']
    cluster_target = 'csw'
    df = stuff.assign_fuzzy_clusters(df, stuff_features, cluster_target)
    df = prc.save_memory(df)
    df.to_parquet(f'{path}clustered_pitch_data.parquet')

    # Location Distribution Creation
    cluster_dist = stuff.create_location_distributions(df)
    count_frequencies = stuff.calculate_count_frequencies(df)
    platoon_cluster_dist = stuff.combine_flatten_distributions(cluster_dist, count_frequencies)
    joblib.dump(platoon_cluster_dist, 'data/platoon_cluster_dist.dat')

    # Model Training
    feat = ['speed', 'vert_approach_angle_adj', 'transverse', 'transverse_pit', 
            'transverse_bat', 'lift', 'release_pos_x', 'release_pos_x_pit',
            'release_pos_x_bat', 'release_pos_y', 'release_pos_z', 'plate_x', 
            'plate_x_pit','plate_x_bat', 'plate_x_abs', 'plate_z', 'plate_z_top', 
            'plate_z_bot', 'plate_dist', 'balls', 'strikes', 'speed_diff', 'lift_diff', 
            'transverse_pit_diff', 'vert_approach_angle', 'game_year']
    model_filters = stuff.get_model_filters(df)
    xgb_models = stuff.train_model(df, model_filters, feat)
    joblib.dump(xgb_models, f'{path}xgb_models.dat')

    # Simulation
    batch_size = 10000
    n_batches = 3
    rv = pd.read_csv('runvalue.csv')
    stuff.simulate_pitches(df, batch_size, n_batches, platoon_cluster_dist, feat, rv, xgb_models, path, year=year)

    # Collect Simulation Results
    simR = []
    simL = []
    for i in range(n_batches):
        simR.append(pd.read_parquet(f'data/sim_vsR_batch{i+1}.parquet'))
        simL.append(pd.read_parquet(f'data/sim_vsL_batch{i+1}.parquet'))
    simR = pd.concat(simR).reset_index(drop=True)
    simL = pd.concat(simL).reset_index(drop=True)
    simR.to_parquet(f'{path}sim_vsR.parquet')
    simL.to_parquet(f'{path}sim_vsL.parquet')

    # Distillation and Prediction
    distill_features = ['speed', 'speed_diff', 'lift', 'lift_diff', 'transverse', 
                        'transverse_pit', 'transverse_pit_diff', 'release_pos_x', 
                        'release_pos_x_pit', 'release_pos_y', 'release_pos_z', 
                        'vert_approach_angle_adj']
    events = [x for x in simR.columns if x.startswith('x_')]
    vsR_models = stuff.train_distilled_models(simR, distill_features, events)
    vsL_models = stuff.train_distilled_models(simL, distill_features, events)
    df = stuff.make_distilled_predictions(df, distill_features, events, vsR_models, vsL_models, game_year=year)
    df.to_parquet(f'{path}pitch_data_with_predictions.parquet')

    # Generate Results
    stuff.generate_results_csv(df, distill_features)

# Call the pipeline
path = 'test/'
pipeline(path, year=2022)