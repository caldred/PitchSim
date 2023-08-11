import pandas as pd
import numpy as np
import skfuzzy as fuzz

from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from xgboost import XGBRegressor

from tune_xgboost import tune_xgboost, distill_params

grid_x = np.linspace(-2.7, 2.7, 28)
grid_z = np.linspace(-1, 5.4, 33)
grid = np.array(np.meshgrid(grid_x, grid_z)).T.reshape(-1,2)
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

def define_cluster_type(df):
    cluster_type = dict(df.groupby('cluster').pitch_type.agg(pd.Series.mode))

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


def assign_fuzzy_clusters(df, features, target):

    xgb1 = tune_xgboost(df, target, features=features)
    f = xgb1.get_booster().get_score(importance_type='total_gain')

    scaler = StandardScaler()
    sdf = pd.DataFrame(scaler.fit_transform(df[features]))
    sdf.columns = features
    for col in f:
        sdf[col] *= f[col]/max(f.values())

    cols = ['speed_diff', 'lift_diff', 'transverse_pit_diff']
    sdf[cols] = sdf[cols].fillna(0)

    pca = PCA(n_components=6)
    sdf_pca = pca.fit_transform(sdf)

    while True:
        cntr, _, _, _, _, _, _ = fuzz.cluster.cmeans(sdf_pca[np.random.choice(a=len(sdf_pca), size=100000)].T, 
                                                     c=8, m=1.5, error=1e-3, maxiter=100)
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(test_data=sdf_pca.T, cntr_trained=cntr, 
                                                       m=1.5, error=1e-3, maxiter=100)
        u = pd.DataFrame(u.T)
        u = u.rename(columns={0:'cluster1', 1:'cluster2', 2:'cluster3', 3:'cluster4',
                              4:'cluster5', 5:'cluster6', 6:'cluster7', 7:'cluster8'})

        cols = ['pitch_type', 'transverse_pit', 'release_pos_z'] + [x for x in df.columns if 'cluster' in x]
        temp = pd.concat([df[cols], u], axis=1)
        temp['cluster'] = temp[[x for x in temp.columns if 'cluster' in x]].idxmax(axis=1)
        cluster_type = define_cluster_type(temp)
        print(cluster_type)
        if len(set(cluster_type.values())) == 8:
            del temp
            break
    df = pd.concat([df, u], axis=1)
    df = df.rename(columns=cluster_type)
    df['cluster'] = df[cluster_names].idxmax(axis=1)

    return df

def kde_cluster(d, cluster):
    group_cols = ['stand', 'p_throws', 'balls', 'strikes']
    kde = KernelDensity(bandwidth = 0.1)
    sample_ = d[['plate_x', 'plate_z']].sample(n=100000, weights=d[cluster], replace=True)
    kde.fit(sample_.values)
    return {tuple(d[group_cols].mode().values[0]): kde.score_samples(grid)}

def create_location_distributions(df):
    group_cols = ['stand', 'p_throws', 'balls', 'strikes']
    cols =  ['plate_x', 'plate_z'] + group_cols + cluster_names
    grouped = df[cols].groupby(group_cols)
    dfs = [grouped.get_group(x) for x in grouped.groups]

    cluster_dist = {}
    for cluster in cluster_names:
        cluster_dist[cluster] = Parallel(n_jobs=10)(
            delayed(kde_cluster)(df_, cluster) for df_ in dfs
        )
        cluster_dist[cluster] = {k: v for d in cluster_dist[cluster] for k, v in d.items()}

    return cluster_dist

def calculate_count_frequencies(df):
    count_frequencies = {}
    for p_throws in ['R', 'L']:
        for stand in ['R', 'L']:
            df_hand = df[(df['p_throws'] == p_throws) & (df['stand'] == stand)]
            count_frequencies[(p_throws, stand)] = {}
            for cluster in cluster_names:
                count_frequencies[(p_throws, stand)][cluster] = {}
                for count in counts:
                    balls, strikes = count
                    count_df = df_hand[(df_hand['balls'] == balls) & (df_hand['strikes'] == strikes)]
                    frequency = count_df[cluster].mean()
                    count_frequencies[(p_throws, stand)][cluster][count] = frequency
    
    return count_frequencies

def combine_flatten_distributions(cluster_dist, count_frequencies):
    dist = {}
    for cluster in cluster_names:
        dist[('R', 'R', cluster)] = []
        dist[('R', 'L', cluster)] = []
        dist[('L', 'R', cluster)] = []
        dist[('L', 'L', cluster)] = []
        for count in counts:
            for bat_side in ['R', 'L']:
                for pitch_hand in ['R', 'L']:
                    cluster_key = (bat_side, pitch_hand, count[0], count[1])
                    log_prob = cluster_dist[cluster].get(cluster_key, -np.inf)
                    prob_weight = np.exp(log_prob)
                    count_weight = count_frequencies[(bat_side, pitch_hand)][cluster][count]
                    dist[(bat_side, pitch_hand, cluster)].append(prob_weight * count_weight)
        for bat_side in ['R', 'L']:
            for pitch_hand in ['R', 'L']:
                dist[(bat_side, pitch_hand, cluster)] = np.array(dist[(bat_side, pitch_hand, cluster)]).reshape(-1)
                dist[(bat_side, pitch_hand, cluster)] /= sum(dist[(bat_side, pitch_hand, cluster)])

    return dist

def get_model_filters(df):
    model_filters = {}
    model_filters['swing'] = (df.swing >= 0)
    model_filters['contact'] = (df.swing == 1)
    model_filters['callstr'] = (df.swing == 0)
    model_filters['hbp'] = (df.nostrike == 1)
    model_filters['inplay'] = (df.contact == 1)
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

def train_models(df, model_filters, features):
    xgb_models = {}

    for target in model_filters.keys():
        df1 = df[model_filters[target]].copy()
        xgb_models[target] = tune_xgboost(df1, target, features)

    return xgb_models

def simulate_pitches(df, batch_size, n_batches, dist, features, rv, xgb_models, path, year=None):

    grid_len = len(grid)
    count_len = len(counts)

    sz_top_avg = df.sz_top.mean()
    sz_bot_avg = df.sz_bot.mean()
    sz_mid_avg = (sz_top_avg + sz_bot_avg)/2

    targets = list(xgb_models.keys())

    for i in range(n_batches):
        if year is not None:
            ds = df[df.game_year==year].sample(batch_size).copy()
        else:
            ds = df.sample(batch_size).copy()
        
        res = {}
        res['R'] = []
        res['L'] = []

        for index, pitch in ds.iterrows():
            print(index)
            for bat_side in ['R', 'L']:
                stand_r = 1 if bat_side == 'R' else 0
                throws_r = 1 if pitch.p_throws == 'R' else 0

                p = np.zeros(len(dist[(bat_side, pitch.p_throws, cluster_names[0])]))
                
                for cluster, weight in zip(cluster_names, pitch[cluster_names]):
                    p += dist[(bat_side, pitch.p_throws, cluster)] * weight

                balls = []
                strikes = []
                for count in counts:
                    balls.append(count[0])
                    strikes.append(count[1])

                balls = np.array(balls)
                strikes = np.array(strikes)

                sim = pd.DataFrame(np.repeat([pitch[features].values], len(counts)*len(grid), axis=0),
                                                columns=features).infer_objects()
                sim['balls'] = np.repeat(balls, grid_len)
                sim['strikes'] = np.repeat(strikes, grid_len)
                sim['plate_x'] = np.tile(grid[:, 0], count_len)
                sim['plate_z'] = np.tile(grid[:, 1], count_len)
                sim['plate_x_abs'] = np.abs(sim.plate_x)
                sim['plate_x_pit'] = sim.plate_x*(2*throws_r-1)
                sim['plate_x_bat'] = sim.plate_x*(2*stand_r-1)
                sim['plate_z_top'] = sim.plate_z - sz_top_avg
                sim['plate_z_bot'] = sim.plate_z - sz_bot_avg
                sim['plate_dist'] = np.sqrt(np.square(sim.plate_x) + np.square(sim.plate_z-sz_mid_avg))

                if 'transverse_bat' in features:
                    sim['transverse_bat'] = sim.transverse*(2*stand_r-1)
                if 'release_pos_x_bat' in features:
                    sim['release_pos_x_bat'] = sim.release_pos_x*(2*stand_r-1)
                if 'vert_approach_angle' in features:
                    sim['vert_approach_angle'] = sim.vert_approach_angle_adj - sim.plate_z/sim.release_pos_y
                
                sim['p'] = p

                sim = sim.merge(rv, on=['balls', 'strikes'], how='inner')


                def get_predictions(df, model, target):
                    return {target: model.predict_proba(df[features])[:, 1]}

                results = Parallel(n_jobs=-1)(delayed(get_predictions)(sim, xgb_models[target], target) for target in targets)

                predictions = {k: v for result in results for k, v in result.items()}

                for target in targets:
                    sim[target] = predictions[target]

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
                
                res[bat_side].append(sim[['x_' + event for event in events + ['run_value']]].sum())

        dsr = pd.concat([ds.reset_index(drop=True), pd.DataFrame(res['R'])], axis=1)
        dsl = pd.concat([ds.reset_index(drop=True), pd.DataFrame(res['L'])], axis=1)
        dsr.to_parquet(f"{path}sim_vsR_batch{i+1}.parquet")
        dsl.to_parquet(f"{path}sim_vsL_batch{i+1}.parquet")

    return sim
    
def train_distilled_models(df, features, events):
    event_xgbs = {}
    for event in events:
        event_xgbs[event] = tune_xgboost(df, event, features, param_dist=distill_params, max_evals=10,
                                         model=XGBRegressor, scoring='neg_mean_squared_error')
    
    return event_xgbs

def make_distilled_predictions(df, distill_features, events, vsR_models, vsL_models, game_year=None):
    for event in events:
        print(event)
        df[event + '_vsR'] = vsR_models[event].predict(df[distill_features])
        df[event + '_vsL'] = vsL_models[event].predict(df[distill_features])

    event_sum_R = df[[event + '_vsR' for event in events if event != 'x_run_value']].sum(axis=1)
    event_sum_L = df[[event + '_vsL' for event in events if event != 'x_run_value']].sum(axis=1)
    for event in events:
        if event != 'x_run_value':
            df[event + '_vsR'] /= event_sum_R
            df[event + '_vsL'] /= event_sum_L
    if game_year is None:
        vsR_avg = df.x_run_value_vsR.mean()
        vsL_avg = df.x_run_value_vsL.mean()
    else:
        vsR_avg = df[df.game_year == game_year].x_run_value_vsR.mean()
        vsL_avg = df[df.game_year == game_year].x_run_value_vsL.mean()
    df['x_run_value_vsR'] -= vsR_avg
    df['x_run_value_vsL'] -= vsL_avg

    return df

def generate_results_csv(df, distill_features):
    group_cols = ['player_name', 'pitch_type', 'game_year']
    cols = group_cols + distill_features + [x for x in df.columns if x.startswith('x_')]
    grp_filt = df[cols].groupby(by=group_cols).filter(lambda x: len(x) >= 10)
    results = grp_filt[cols].groupby(by=group_cols).mean().dropna().reset_index()
    res_count = grp_filt[cols].groupby(by=group_cols).speed.count()
    res_count = res_count.reset_index()
    res_count = res_count.rename(columns={'speed':'pitches'})
    results  = results.merge(res_count, on=['player_name', 'pitch_type', 'game_year'])
    results['speed'] *= 0.681818
    results['speed_diff'] *= 0.681818
    results['xRV100_vsR'] = results.x_run_value_vsR*100
    results['xRV100_vsL'] = results.x_run_value_vsL*100
    results['xRV100'] = results.xRV100_vsR*0.6 + results.xRV100_vsL*0.4
    results[['player_name', 'pitch_type', 'game_year', 'pitches', 'xRV100_vsR', 'xRV100_vsL', 'xRV100'] + 
            [x for x in df.columns if (x.startswith('x_'))] + distill_features].to_csv('results.csv', index=False)
