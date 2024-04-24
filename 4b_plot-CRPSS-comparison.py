# author: Thibault Hallouin, INRAE
# date: June 2023
# description: plot comparison between evalhyd results and Harrigan et al. (2023) results

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


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

# read in evalhyd results
hallouin_persistence = np.genfromtxt(
    "output/CRPSS-persistence.csv", delimiter=','
)
hallouin_climatology = np.genfromtxt(
    "output/CRPSS-climatology.csv", delimiter=','
)

# collect GloFAS station meta data
meta = pd.read_csv('data/Harrigan2023-GloFAS-river_points.csv')

# gather results from Harrigan et al. (2023)
harrigan_persistence = np.zeros((prd.station.size, prd.step.size))
harrigan_persistence[:] = np.nan
harrigan_climatology = np.zeros((prd.station.size, prd.step.size))
harrigan_climatology[:] = np.nan

data_prs = pd.read_csv('data/Harrigan2023-GloFAS-CRPSS_persistence.csv')
data_clm = pd.read_csv('data/Harrigan2023-GloFAS-CRPSS_climatology.csv')

for s in range(prd.station.size):
    id_, = meta.loc[:, ['GloFAS_ID']].iloc[s]

    harrigan_persistence[s, :] = (
        data_prs.loc[data_prs['GloFAS_ID'] == id_].loc[
            :, ['LT1', 'LT2', 'LT3', 'LT4', 'LT5', 'LT6', 'LT7', 'LT8', 'LT9',
                'LT10', 'LT12', 'LT14', 'LT16', 'LT18', 'LT20', 'LT25', 'LT30']
        ]
    )

    harrigan_climatology[s, :] = (
        data_clm.loc[data_clm['GloFAS_ID'] == id_].loc[
            :, ['LT1', 'LT2', 'LT3', 'LT4', 'LT5', 'LT6', 'LT7', 'LT8', 'LT9',
                'LT10', 'LT12', 'LT14', 'LT16', 'LT18', 'LT20', 'LT25', 'LT30']
        ]
    )

# -----------------------------------------------------------------------------
# plot numerical comparison for persistence and climatology
# -----------------------------------------------------------------------------
for benchmark_name, hallouin_data, harrigan_data in zip(
        ('persistence', 'climatology'),
        (hallouin_persistence, hallouin_climatology),
        (harrigan_persistence, harrigan_climatology)
):
    fig, axs = plt.subplots(nrows=5, ncols=6, layout="compressed", figsize=(8, 8))
    gs = axs[4, 3].get_gridspec()

    ldt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]

    for s, ax in enumerate(axs.flat):
        if s < prd.station.size:
            id_, = meta.loc[:, ['GloFAS_ID']].iloc[s]

            ax.set_aspect(1)

            ax.scatter(
                x=harrigan_data[s, :],
                y=np.around(hallouin_data[s, :], 2),
                c=ldt,
                zorder=9
            )

            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            ax_min, ax_max = min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])
            ax.set_xlim(ax_min, ax_max)
            ax.set_ylim(ax_min, ax_max)

            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            ax.axline(
                (0, 0), slope=1,
                color="black", linewidth=mpl.rcParams['axes.linewidth'],
                zorder=1
            )

            ax.set_title(id_, fontsize=mpl.rcParams['xtick.labelsize'])
        else:
            ax.remove()

    gs = mpl.gridspec.GridSpec(5, 60, figure=fig)
    cax = fig.add_subplot(gs[4, 32:59])
    img = cax.imshow(np.array([ldt]))
    cbar = plt.colorbar(img, cax=cax, cmap='viridis', orientation='horizontal')
    cbar.set_ticks(ldt)
    cbar.set_label("lead time [days]")

    fig.supxlabel('CRPSS as reported by Harrigan et al. (2023)')
    fig.supylabel(r'CRPSS reproduced using $\mathtt{evalhyd}$')

    plt.rcParams.update(
        {
            "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
            "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
            "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)
        }
    )

    fig.savefig(f"output/CRPSS-{benchmark_name}-diag.png", dpi=300)
