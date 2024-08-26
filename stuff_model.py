import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from tune_xgboost import tune_xgboost, distill_params
from processing import save_memory
from typing import List, Dict, Tuple, Optional

# Predefined constants for grid dimensions and events
grid_x = np.linspace(-2.7, 2.7, 28)
grid_z = np.linspace(-1, 5.4, 33)
grid = np.array(np.meshgrid(grid_x, grid_z)).T.reshape(-1, 2)
counts = [(balls, strikes) for balls in range(4) for strikes in range(3)]
events = ['callstr', 'ball', 'hbp', 'swstr', 'foul', 'pop', 'hr', 'if1b', 'gbout', 'gidp', 'air_out', '1b', '2b', '3b']
cluster_names = [
    'sinker',
    'low-slot fastball',
    'gyro slider',
    'sweeper',
    'offspeed',
    'curveball',
    'high-slot fastball',
    'cutter'
]

def define_cluster_type(df: pd.DataFrame) -> Dict[int, str]:
    """
    Determine the type of cluster based on its characteristics.

    Parameters:
    - df (pd.DataFrame): The data frame containing pitch data.

    Returns:
    - Dict[int, str]: A dictionary mapping cluster IDs to their type.
    """

    # Group by cluster and get the mode of pitch type for each cluster
    cluster_type = dict(df.groupby('cluster').pitch_type.agg(pd.Series.mode))
    
    # Map the resulting pitch types to their corresponding cluster types
    for key in cluster_type:
        if cluster_type[key] == 'CH':
            cluster_type[key] = 'offspeed'
        elif cluster_type[key] in ['CU', 'SV']:
            cluster_type[key] = 'curveball'
        elif cluster_type[key] == 'FC':
            cluster_type[key] = 'cutter'
        elif cluster_type[key] == 'SI':
            cluster_type[key] = 'sinker'
        elif cluster_type[key] == 'FF':
            if df[df.cluster == key].release_pos_z.mean() < 6:
                cluster_type[key] = 'low-slot fastball'
            else:
                cluster_type[key] = 'high-slot fastball'
        elif cluster_type[key] in ['SL', 'ST']:
            if df[df.cluster == key].transverse_pit.mean()  > -5.5:
                cluster_type[key] = 'gyro slider'
            else:
                cluster_type[key] = 'sweeper'
    
    return cluster_type

def assign_fuzzy_clusters(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    """
    Assign clusters to data points using fuzzy clustering and returns the data frame with cluster assignments.

    Parameters:
    - df (pd.DataFrame): The data frame containing the features.
    - features (List[str]): The columns in the data frame to be used as features.
    - target (str): The target variable.

    Returns:
    - pd.DataFrame: The data frame with cluster assignments added.
    """

    # Start by tuning the XGBoost model
    xgb1 = tune_xgboost(df, target, features=features)
    
    # Get the feature importance
    f = xgb1.get_booster().get_score(importance_type='total_gain')

    # Standardize the features
    scaler = StandardScaler()
    sdf = pd.DataFrame(scaler.fit_transform(df[features]))
    sdf.columns = features

    # Adjust standardized values by their importance
    for col in f:
        sdf[col] *= f[col]/max(f.values())
    
    # Handle missing values in specific columns
    cols = ['speed_diff', 'lift_diff', 'transverse_pit_diff']
    sdf[cols] = sdf[cols].fillna(0)

    # Reduce dimensionality using PCA
    pca = PCA(n_components=6)
    sdf_pca = pca.fit_transform(sdf)

    # Use fuzzy clustering to assign clusters to data points
    while True:
        # Apply the c-means clustering algorithm
        cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(
            sdf_pca[np.random.choice(a=len(sdf_pca), size=100000)].T, 
            c=8, m=1.5, error=1e-3, maxiter=100
        )

        # Predict the cluster memberships for the data points
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            test_data=sdf_pca.T, cntr_trained=cntr, 
            m=1.5, error=1e-3, maxiter=100
        )
        u = pd.DataFrame(u.T)
        u = u.rename(columns={0:'cluster1', 1:'cluster2', 2:'cluster3', 3:'cluster4',
                              4:'cluster5', 5:'cluster6', 6:'cluster7', 7:'cluster8'})

        # Assign cluster types
        cols = ['pitch_type', 'transverse_pit', 'release_pos_z'] + [x for x in df.columns if 'cluster' in x]
        temp = pd.concat([df[cols], u], axis=1)
        temp['cluster'] = temp[[x for x in temp.columns if 'cluster' in x]].idxmax(axis=1)
        cluster_type = define_cluster_type(temp)
        print(cluster_type)

        # Break the loop when all cluster types are unique
        if len(set(cluster_type.values())) == 8:
            del temp
            break

    # Merge the original dataframe with cluster data
    df = pd.concat([df, u], axis=1)
    df = df.rename(columns=cluster_type)
    df['cluster'] = df[cluster_names].idxmax(axis=1)

    return df

def kde_cluster(d: pd.DataFrame, cluster: str) -> Dict[Tuple[str, str, int, int], np.ndarray]:
    """
    Compute Kernel Density Estimation (KDE) for a given cluster.
    
    Args:
    - d (pd.DataFrame): Subset of the data for a specific group.
    - cluster (str): Name of the cluster for which KDE is being computed.
    
    Returns:
    - Dict[Tuple[str, str, int, int], np.ndarray]: Dictionary with group tuple as key and KDE scores as values.
    """
    group_cols = ['stand', 'p_throws', 'balls', 'strikes']
    kde = KernelDensity(bandwidth=0.1)
    
    # Sample data for KDE
    sample_ = d[['plate_x', 'plate_z']].sample(n=100000, weights=d[cluster], replace=True)
    kde.fit(sample_.values)
    
    return {tuple(d[group_cols].mode().values[0]): kde.score_samples(grid)}

def create_location_distributions(df: pd.DataFrame) -> Dict[str, Dict[Tuple[str, str, int, int], np.ndarray]]:
    """
    Cluster the dataframe based on pitch location and compute the KDE for each cluster.
    
    Args:
    - df (pd.DataFrame): Input dataframe with pitch data.
    
    Returns:
    - Dict[str, Dict[Tuple[str, str, int, int], np.ndarray]]: KDE distributions for each cluster.
    """
    group_cols = ['stand', 'p_throws', 'balls', 'strikes']
    cols =  ['plate_x', 'plate_z'] + group_cols + cluster_names
    grouped = df[cols].groupby(group_cols)
    dfs = [grouped.get_group(x) for x in grouped.groups]

    cluster_dist = {}
    
    # Compute KDE for each cluster in parallel
    for cluster in cluster_names:
        cluster_dist[cluster] = Parallel(n_jobs=10)(
            delayed(kde_cluster)(df_, cluster) for df_ in dfs
        )
        cluster_dist[cluster] = {k: v for d in cluster_dist[cluster] for k, v in d.items()}
        
    return cluster_dist

def calculate_count_frequencies(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Dict[Tuple[int, int], float]]]:
    """
    Compute the frequency of each pitch type for different counts and handedness combinations.
    
    Args:
    - df (pd.DataFrame): Input dataframe with pitch data.
    
    Returns:
    - Dict[Tuple[str, str], Dict[str, Dict[Tuple[int, int], float]]]: Frequency distributions for each scenario.
    """
    count_frequencies = {}
    
    # Loop over pitcher and batter handedness
    for p_throws in ['R', 'L']:
        for stand in ['R', 'L']:
            df_hand = df[(df['p_throws'] == p_throws) & (df['stand'] == stand)]
            count_frequencies[(p_throws, stand)] = {}
            
            # Compute frequency for each cluster and count
            for cluster in cluster_names:
                count_frequencies[(p_throws, stand)][cluster] = {}
                for count in counts:
                    balls, strikes = count
                    count_df = df_hand[(df_hand['balls'] == balls) & (df_hand['strikes'] == strikes)]
                    frequency = count_df[cluster].mean()
                    count_frequencies[(p_throws, stand)][cluster][count] = frequency
    
    return count_frequencies

def combine_flatten_distributions(cluster_dist: Dict[str, Dict[Tuple[str, str, int, int], np.ndarray]], 
                                  count_frequencies: Dict[Tuple[str, str], Dict[str, Dict[Tuple[int, int], float]]]) -> Dict[Tuple[str, str, str], np.ndarray]:
    """
    Combine and flatten the pitch distributions based on location and count frequencies.
    
    Args:
    - cluster_dist (Dict[str, Dict[Tuple[str, str, int, int], np.ndarray]]): KDE distributions for each cluster.
    - count_frequencies (Dict[Tuple[str, str], Dict[str, Dict[Tuple[int, int], float]]]): Frequency distributions for each scenario.
    
    Returns:
    - Dict[Tuple[str, str, str], np.ndarray]: Combined distributions.
    """
    dist = {}
    
    # Combine location and count distributions
    for cluster in cluster_names:
        for bat_side in ['R', 'L']:
            for pitch_hand in ['R', 'L']:
                dist_key = (bat_side, pitch_hand, cluster)
                dist[dist_key] = []
                
                for count in counts:
                    balls, strikes = count
                    cluster_key = (bat_side, pitch_hand, balls, strikes)
                    
                    # Get log probability of the cluster for the current count. Default to negative infinity if not found.
                    log_prob = cluster_dist[cluster].get(cluster_key, -np.inf)
                    
                    # Convert log probability to probability
                    prob_weight = np.exp(log_prob)
                    
                    # Get frequency of the current cluster for the current count and handedness
                    count_weight = count_frequencies[(bat_side, pitch_hand)][cluster][count]
                    
                    # Combine location-based and count-based distributions
                    dist[dist_key].append(prob_weight * count_weight)
                
                # Normalize the combined distributions
                dist[dist_key] = np.array(dist[dist_key]).reshape(-1)
                dist[dist_key] /= sum(dist[dist_key])

    return dist

def get_model_filters(df: pd.DataFrame) -> dict:
    """
    Define filters for different event outcomes in a baseball game.

    Parameters:
    - df (pd.DataFrame): The data containing event outcomes.

    Returns:
    - dict: A dictionary containing boolean masks for each event outcome.
    """
    
    model_filters = {}
    
    # Filters for different pitch outcomes
    model_filters['swing'] = (df.swing >= 0)
    model_filters['contact'] = (df.swing == 1)
    model_filters['callstr'] = (df.swing == 0)
    model_filters['hbp'] = (df.nostrike == 1)
    model_filters['inplay'] = (df.contact == 1)

    # Filters for different types of hits
    model_filters['air'] = (df.inplay == 1)
    model_filters['pop'] = (df.air == 1)
    model_filters['ofgb'] = (df.gb == 1)
    model_filters['hr'] = (df.fbld == 1)
    model_filters['if1b'] = (df.ifgb == 1)
    model_filters['gidp'] = (df.gb_out == 1) & (df.is_on_1b == 1) & (df.outs_when_up < 2)
    model_filters['air_hit'] = (df.fbld_inplay == 1)
    model_filters['xbh'] = (df.air_hit == 1) | (df.ofgb == 1)
    model_filters['triple'] = (df.xbh == 1)

    return model_filters

def train_models(df: pd.DataFrame, model_filters: dict, features: list) -> dict:
    """
    Train models for different event outcomes using the provided filters.

    Parameters:
    - df (pd.DataFrame): The data containing event outcomes.
    - model_filters (dict): Filters defined for each event outcome.
    - features (list): List of feature column names used for training.

    Returns:
    - dict: Dictionary containing trained models for each event outcome.
    """
    
    xgb_models = {}

    # Train a model for each event outcome using the provided filters
    for target in model_filters.keys():
        df1 = df[model_filters[target]].copy()
        xgb_models[target] = tune_xgboost(df1, target, features)

    return xgb_models

def simulate_pitches(df: pd.DataFrame, batch_size: int, n_batches: int, dist: dict, 
                     features: list, rv: pd.DataFrame, xgb_models: dict, 
                     path: str, year: Optional[int] = None) -> pd.DataFrame:
    """
    Simulate the outcome of pitches based on the trained models and the provided distributions.
    
    Parameters:
    - df (pd.DataFrame): Input data.
    - batch_size (int): Number of rows to sample in each batch.
    - n_batches (int): Number of batches to simulate.
    - dist (dict): Probabilities for different clusters.
    - features (list): List of feature column names used for prediction.
    - rv (pd.DataFrame): Run values for different pitch outcomes.
    - xgb_models (dict): Trained models for different pitch outcomes.
    - path (str): Path to save the simulated data.
    - year (Optional[int]): Year to filter the data on. If None, data isn't filtered on year.
    
    Returns:
    - pd.DataFrame: Simulated pitch outcomes.
    """

    # Define constants used for simulation
    grid_len = len(grid)
    count_len = len(counts)
    sz_top_avg = df.sz_top.mean()
    sz_bot_avg = df.sz_bot.mean()
    sz_mid_avg = (sz_top_avg + sz_bot_avg) / 2
    targets = list(xgb_models.keys())

    # Start simulating for each batch
    for i in range(n_batches):
        if year is not None:
            ds = df[df.game_year == year].sample(batch_size).copy()
        else:
            ds = df.sample(batch_size).copy()
        
        res = {'R': [], 'L': []}

        # Simulate for each pitch in the dataset
        for index, pitch in ds.iterrows():
            for bat_side in ['R', 'L']:
                stand_r = 1 if bat_side == 'R' else 0
                throws_r = 1 if pitch.p_throws == 'R' else 0

                # Initialize a probability array
                p = np.zeros(len(dist[(bat_side, pitch.p_throws, cluster_names[0])]))

                # Calculate probabilities based on cluster weights
                for cluster, weight in zip(cluster_names, pitch[cluster_names]):
                    p += dist[(bat_side, pitch.p_throws, cluster)] * weight

                # Define balls and strikes from counts
                balls = np.array([count[0] for count in counts])
                strikes = np.array([count[1] for count in counts])

                # Create a dataframe for simulation
                sim = pd.DataFrame(np.repeat([pitch[features].values], len(counts) * len(grid), axis=0),
                                   columns=features)
                sim['balls'] = np.repeat(balls, grid_len)
                sim['strikes'] = np.repeat(strikes, grid_len)
                sim['plate_x'] = np.tile(grid[:, 0], count_len)
                sim['plate_z'] = np.tile(grid[:, 1], count_len)

                # Additional feature engineering based on the bat side and pitch hand
                sim['plate_x_abs'] = np.abs(sim.plate_x)
                sim['plate_x_pit'] = sim.plate_x * (2 * throws_r - 1)
                sim['plate_x_bat'] = sim.plate_x * (2 * stand_r - 1)
                sim['plate_z_top'] = sim.plate_z - sz_top_avg
                sim['plate_z_bot'] = sim.plate_z - sz_bot_avg
                sim['plate_dist'] = np.sqrt(np.square(sim.plate_x) + np.square(sim.plate_z - sz_mid_avg))

                if 'transverse_bat' in features:
                    sim['transverse_bat'] = sim.transverse * (2 * stand_r - 1)
                if 'release_pos_x_bat' in features:
                    sim['release_pos_x_bat'] = sim.release_pos_x * (2 * stand_r - 1)
                if 'vert_approach_angle' in features:
                    sim['vert_approach_angle'] = sim.vert_approach_angle_adj - sim.plate_z / sim.release_pos_y

                sim['p'] = p

                # Merge with run value data
                sim = sim.merge(rv, on=['balls', 'strikes'], how='inner')

                for feature in features:
                    sim[feature] = sim[feature].astype(float)

                # Predict outcomes using the trained models 
                def get_predictions(sim, model, target):
                    return {target: model.predict_proba(sim[features])[:, 1]}

                results = Parallel(n_jobs=-1)(delayed(get_predictions)(sim, xgb_models[target], target) for target in targets)

                predictions = {k: v for result in results for k, v in result.items()}

                for target in targets:
                    sim[target] = predictions[target]

                # Calculate expected outcomes and run values
                sim['x_callstr'] = (1-sim.swing)*sim.callstr
                sim['x_ball'] = (1-sim.swing)*(1-sim.callstr)*(1-sim.hbp)
                sim['x_hbp'] = (1-sim.swing)*(1-sim.callstr)*sim.hbp
                sim['x_swstr'] = sim.swing*(1-sim.contact)
                sim['x_foul'] = sim.swing*sim.contact*(1-sim.inplay)
                sim['x_pop'] = sim.swing*sim.contact*sim.inplay*sim.air*sim['pop']
                sim['x_hr'] = sim.swing*sim.contact*sim.inplay*sim.air*(1-sim['pop'])*sim.hr
                sim['x_if1b'] = sim.swing*sim.contact*sim.inplay*(1-sim.air)*(1-sim.ofgb)*sim.if1b
                sim['x_gbout'] = sim.swing*sim.contact*sim.inplay*(1-sim.air)*(1-sim.ofgb)*(1-sim.if1b)
                sim['x_gidp'] = sim.swing*sim.contact*sim.inplay*(1-sim.air)*(1-sim.ofgb)*(1-sim.if1b)*sim.gidp
                sim['x_air_out'] = sim.swing*sim.contact*sim.inplay*sim.air*(1-sim['pop'])*(1-sim.hr)*(1-sim.air_hit)
                sim['x_1b'] =  sim.swing*sim.contact*sim.inplay*(sim.air*(1-sim['pop'])*(1-sim.hr)*sim.air_hit + (1-sim.air)*sim.ofgb)*(1-sim.xbh)
                sim['x_2b'] =  sim.swing*sim.contact*sim.inplay*(sim.air*(1-sim['pop'])*(1-sim.hr)*sim.air_hit + (1-sim.air)*sim.ofgb)*sim.xbh*(1-sim.triple)
                sim['x_3b'] =  sim.swing*sim.contact*sim.inplay*(sim.air*(1-sim['pop'])*(1-sim.hr)*sim.air_hit + (1-sim.air)*sim.ofgb)*sim.xbh*sim.triple

                for event in events:
                    sim[f'x_{event}'] *= p

                sim['x_run_value'] = (sim.x_callstr + sim.x_swstr)*sim.rv_strike + sim.x_ball*sim.rv_ball + sim.x_hbp*sim.rv_hbp + sim.x_foul*sim.rv_foul
                sim['x_run_value'] += (sim.x_pop + sim.x_gbout + sim.x_air_out)*sim.rv_out + (sim.x_if1b + sim.x_1b)*sim.rv_single
                sim['x_run_value'] += sim.x_2b*sim.rv_double + sim.x_3b*sim.rv_triple + sim.x_hr*sim.rv_home_run
                
                # Store the results
                res[bat_side].append(sim[['x_' + event for event in events + ['run_value']]].sum())

        # Save the simulation results
        dsr = pd.concat([ds.reset_index(drop=True), pd.DataFrame(res['R'])], axis=1)
        dsl = pd.concat([ds.reset_index(drop=True), pd.DataFrame(res['L'])], axis=1)
        dsr.to_parquet(f"{path}sim_vsR_batch{i+1}.parquet")
        dsl.to_parquet(f"{path}sim_vsL_batch{i+1}.parquet")

    return sim
    
def train_distilled_models(df: pd.DataFrame, 
                           features: List[str], 
                           events: List[str]) -> Dict[str, XGBRegressor]:
    """
    Train models for a list of events using a distilled XGBoost regressor.
    
    Parameters:
    - df (pd.DataFrame): Input data.
    - features (List[str]): List of feature columns.
    - events (List[str]): List of events to train models for.
    
    Returns:
    - Dict[str, XGBRegressor]: Dictionary of trained XGBRegressor models for each event.
    """
    event_xgbs = {}
    for event in events:
        event_xgbs[event] = tune_xgboost(df, event, features, param_dist=distill_params, max_evals=10,
                                         model=XGBRegressor, scoring='neg_mean_squared_error')
    
    return event_xgbs

def make_distilled_predictions(df: pd.DataFrame, 
                               distill_features: List[str], 
                               events: List[str], 
                               vsR_models: Dict[str, XGBRegressor], 
                               vsL_models: Dict[str, XGBRegressor], 
                               game_year: Optional[int] = None) -> pd.DataFrame:
    """
    Make predictions using distilled models for both right and left batting sides.
    
    Parameters:
    - df (pd.DataFrame): Input data.
    - distill_features (List[str]): List of feature columns for distilled models.
    - events (List[str]): List of events.
    - vsR_models (Dict[str, XGBRegressor]): Dictionary of models for right batting side.
    - vsL_models (Dict[str, XGBRegressor]): Dictionary of models for left batting side.
    - game_year (Optional[int]): Specific year for filtering the data. Default is None.
    
    Returns:
    - pd.DataFrame: DataFrame with distilled predictions.
    """
    for event in events:
        print(event)
        df[event + '_vsR'] = vsR_models[event].predict(df[distill_features])
        df[event + '_vsL'] = vsL_models[event].predict(df[distill_features])

    # Normalize the event probabilities so they sum to 1
    event_sum_R = df[[event + '_vsR' for event in events if event != 'x_run_value']].sum(axis=1)
    event_sum_L = df[[event + '_vsL' for event in events if event != 'x_run_value']].sum(axis=1)
    for event in events:
        if event != 'x_run_value':
            df[event + '_vsR'] /= event_sum_R
            df[event + '_vsL'] /= event_sum_L

    # Adjust the run values relative to the average
    if game_year is None:
        vsR_avg = df.x_run_value_vsR.mean()
        vsL_avg = df.x_run_value_vsL.mean()
    else:
        vsR_avg = df[df.game_year == game_year].x_run_value_vsR.mean()
        vsL_avg = df[df.game_year == game_year].x_run_value_vsL.mean()
    df['x_run_value_vsR'] -= vsR_avg
    df['x_run_value_vsL'] -= vsL_avg

    return df

def generate_results_csv(df: pd.DataFrame, 
                         distill_features: List[str]) -> None:
    """
    Generate a CSV file containing aggregated results.
    
    Parameters:
    - df (pd.DataFrame): Input data.
    - distill_features (List[str]): List of feature columns for distilled models.
    
    Returns:
    - None
    """
    group_cols = ['player_name', 'pitch_type', 'game_year']
    cols = group_cols + distill_features + cluster_names + [x for x in df.columns if x.startswith('x_')]

    # Filter out groups with fewer than 10 entries
    grp_filt = df[cols].groupby(by=group_cols, observed=False).filter(lambda x: len(x) >= 10)

    # Aggregate the data
    results = grp_filt[cols].groupby(by=group_cols, observed=False).mean().dropna().reset_index()
    res_count = grp_filt[cols].groupby(by=group_cols, observed=False).speed.count().reset_index().rename(columns={'speed':'pitches'})
    
    results  = results.merge(res_count, on=['player_name', 'pitch_type', 'game_year'])

    # Convert speed to mph and calculate combined run values
    results['speed'] *= 0.681818
    results['speed_diff'] *= 0.681818
    results['xRV100_vsR'] = results.x_run_value_vsR*100
    results['xRV100_vsL'] = results.x_run_value_vsL*100
    results['xRV100'] = results.xRV100_vsR*0.6 + results.xRV100_vsL*0.4

    # Save results to CSV
    results[['player_name', 'pitch_type', 'game_year', 'pitches', 'xRV100_vsR', 'xRV100_vsL', 'xRV100'] + 
            [x for x in df.columns if (x.startswith('x_'))] + distill_features + cluster_names].to_csv('results.csv', index=False)