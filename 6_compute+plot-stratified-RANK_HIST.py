# author: Thibault Hallouin, INRAE
# date: June 2023
# description: compute and plot stratified rank histograms

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

# compute thresholds from climatology quantiles
thr = np.quantile(obs.values, 0.2, axis=1, keepdims=True)

# generate two masks (whole period, low flow period)
cdt = np.repeat(
    np.array(
        [
            ["q_prd_median{<qtl0.3}",
             "q_prd_median{>=qtl0.3,<=qtl0.7}",
             "q_prd_median{>qtl0.7}"]
        ]
    ),
    repeats=prd.station.size, axis=0
)

# evaluate forecast
rank_hist, completeness = evalhyd.evalp(
    q_obs=obs.sel(time=dts).values,
    q_prd=prd_arr,
    metrics=['RANK_HIST'],
    m_cdt=cdt,
    diagnostics=['completeness']
)

# plot
meta = pd.read_csv('data/Harrigan2023-GloFAS-river_points.csv')
sites, = meta.loc[:, ['GloFAS_ID']].values.T
leadtimes = ['LT1', 'LT2', 'LT3', 'LT4', 'LT5', 'LT6', 'LT7', 'LT8', 'LT9',
             'LT10', 'LT12', 'LT14', 'LT16', 'LT18', 'LT20', 'LT25', 'LT30']

fig = plt.figure(figsize=(8, 5), layout="compressed")
gs = mpl.gridspec.GridSpec(1, 3, figure=fig)

s = 3
l = 5

for c, alpha, title in [
        (0, 0.6, r"$\bf{(a)}$ low"),
        (1, 0.8, r"$\bf{(b)}$ average"),
        (2, 1.0, r"$\bf{(c)}$ high")
]:
    ax = fig.add_subplot(gs[:, c])

    ax.bar(
        np.arange(12), rank_hist[s, l, c, 0, :],
        color="tab:blue", alpha=alpha, width=1.0
    )

    ax.set_title(title)

    ax.set_xlim(-0.5, 11.5)
    ax.set_xticks(np.arange(0, 12))
    ax.set_xticklabels(['1', '', '', '', '', '', '', '', '', '', '', '12'])

    ax.set_ylim(0, np.amax(rank_hist[s, l, :, 0, :] * 1.05))
    if c == 0:
        ax.set_ylabel("Observed fraction [-]")
    if c != 0:
        ax.set_yticklabels([])
    if c == 1:
        ax.set_xlabel("Observation rank [-]")

plt.rcParams.update(
    {
        "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
        "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)
    }
)

fig.savefig(
    f"output/RANK_HIST-stratification-{sites[s]}-{leadtimes[l]}.png",
    dpi=300
)
