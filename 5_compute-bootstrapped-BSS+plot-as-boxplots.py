# author: Thibault Hallouin, INRAE
# date: June 2023
# description: compute bootstrapped BSS and display as boxplots

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import evalhyd


def add_station_coord(xd):
    stn = xd.encoding['source'].split('_')[-1].split('.nc')[0]
    xd['dis24'] = xd['dis24'].assign_coords(station=stn)
    return xd


# read in data
prd = xr.open_mfdataset(
    'data/modified/GloFAS-v2.2_river_discharge_reforecast_*.nc',
    preprocess=add_station_coord,
    combine='nested',
    concat_dim='station'
)['dis24']

obs = xr.open_mfdataset(
    'data/modified/GloFAS-ERA5_v2.1_river_discharge_reanalysis_*.nc',
    preprocess=add_station_coord,
    combine='nested',
    concat_dim='station'
)['dis24']

# reorder dimensions follow evalhyd convention
prd = prd.transpose(
    'station',  # site
    'step',  # leadtime
    'number',  # member
    'time'  # time
)
obs = obs.transpose(
    'station',  # site
    'time'  # time
)

# replace time dimension coordinate with valid_time coordinate for
# observations so that time dimension can be used to get temporal
# subsets corresponding to predictions
obs['time'] = obs.valid_time

# determine necessary observations time steps
dts = np.unique(prd.valid_time.values)

# map predictions onto observations time steps
prd_arr = np.zeros((prd.station.size, prd.step.size, prd.number.size, dts.size))
prd_arr[:] = np.nan

for s in range(prd.step.size):
    # get mask selecting time steps where a forecast exists
    msk = np.in1d(dts, prd.valid_time.isel(step=s))
    # use mask to map predictions in array
    prd_arr[:, s, :, :][:, :, msk] = prd.values[:, s, :, :]
    prd_arr[:, s, :, :][:, :, msk] = prd.values[:, s, :, :]

# trim time series to only retain complete years
msk = dts < np.datetime64(f"2019-01-04").astype('datetime64[ns]')
dts = dts[msk]
prd_arr = prd_arr[..., msk].copy()

# compute thresholds from climatology quantiles
thr = np.quantile(obs.values, 0.2, axis=1, keepdims=True)

# evaluate forecast
bss_forecast, = evalhyd.evalp(
    q_obs=obs.sel(time=dts).values,
    q_prd=prd_arr,
    metrics=['BSS'],
    q_thr=thr,
    events='low',
    bootstrap={'n_samples': 1000, 'len_sample': 10, 'summary': 2},
    dts=pd.to_datetime(dts).strftime('%Y-%m-%d %H:%M:%S').values.astype('|S32')
)

# preprocess boxplot stats
stats = np.zeros((prd.station.size, prd.step.size), dtype=dict)

for s in range(prd.station.size):
    for l in range(prd.step.size):
        stats[s][l] = {
            # Bottom whisker position (5th percentile)
            'whislo': bss_forecast[s, l, 0, 0, 0],
            # First quartile (25th percentile)
            'q1': bss_forecast[s, l, 0, 2, 0],
            # Median (50th percentile)
            'med': bss_forecast[s, l, 0, 3, 0],
            # Third quartile (75th percentile)
            'q3': bss_forecast[s, l, 0, 4, 0],
            # Top whisker position (95th percentile)
            'whishi': bss_forecast[s, l, 0, 6, 0],
            # Outliers
            'fliers': []
        }

# collect meta data
meta = pd.read_csv('data/Harrigan2023-GloFAS-river_points.csv')
sites, = meta.loc[:, ['GloFAS_ID']].values.T
leadtimes = ['LT1', 'LT2', 'LT3', 'LT4', 'LT5', 'LT6', 'LT7', 'LT8', 'LT9',
             'LT10', 'LT12', 'LT14', 'LT16', 'LT18', 'LT20', 'LT25', 'LT30']

# plot
lead = 10

fig = plt.figure(figsize=(5, 7), layout="compressed")
gs = mpl.gridspec.GridSpec(100, 100, figure=fig)

ax = fig.add_subplot(gs[:, :])

ax.bxp(
    stats[:, lead],
    vert=False, showbox=True, patch_artist=True,
    boxprops=dict(facecolor='tab:blue', edgecolor='tab:blue'),
    whiskerprops=dict(color='tab:blue'),
    capprops=dict(color='tab:blue'),
    medianprops=dict(color='white')
)

ax.set_xlabel("BSS [-]")
ax.set_yticklabels(sites)

ax.set_facecolor('white')

plt.rcParams.update(
    {
        "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
        "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)
    }
)

fig.savefig(f"output/BSS-boxplots.png", dpi=300)
