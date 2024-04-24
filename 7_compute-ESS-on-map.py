# author: Thibault Hallouin, INRAE
# date: June 2023
# description: compute ESS against climatology benchmark and plot on map

import xarray as xr
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import evalhyd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas
import geoplot


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
prd_arr = np.zeros(
    (prd.station.size, prd.step.size, prd.number.size, dts.size)
)
prd_arr[:] = np.nan

for s in range(prd.step.size):
    # get mask selecting time steps where a forecast exists
    msk = np.in1d(dts, prd.valid_time.isel(step=s))
    # use mask to map predictions in array
    prd_arr[:, s, :, :][:, :, msk] = prd.values[:, s, :, :]
    prd_arr[:, s, :, :][:, :, msk] = prd.values[:, s, :, :]

# compute thresholds from climatology quantiles
thr = np.quantile(obs.values, 0.1, axis=1, keepdims=True)

# collect meta data
meta = pd.read_csv('data/Harrigan2023-GloFAS-river_points.csv')
sites, = meta.loc[:, ['GloFAS_ID']].values.T
leadtimes = [
    'LT1', 'LT2', 'LT3', 'LT4', 'LT5', 'LT6', 'LT7', 'LT8', 'LT9',
    'LT10', 'LT12', 'LT14', 'LT16', 'LT18', 'LT20', 'LT25', 'LT30'
]

# evaluate forecast
basins_to_stations = {
    'Loire': [
        'G3716', 'G3699', 'G3695', 'G3694', 'G1597',
        'G1596', 'G1595', 'G1593', 'G1592', 'G0518'
    ],
    'Garonne': [
        'G1586', 'G1587'
    ],
    'Rhone': [
        'G1585', 'G0554', 'G0565', 'G1591', 'G1594',
        'G1588', 'G1589', 'G1590', 'G3851', 'G1584'
    ],
    'Meuse': [
        'G1601'
    ],
    'Seine': [
        'G1600', 'G1598', 'G1599'
    ],
    'Corse': [
        'G0664'
    ]
}
basins = list(basins_to_stations.keys())

es_prd = np.zeros((len(basins), prd.step.size))
for b, basin in enumerate(basins):
    stn_idx = prd.indexes['station'].get_indexer(basins_to_stations[basin])

    es_prd[b] = evalhyd.evalp(
        q_obs=obs.sel(time=dts).values[stn_idx],
        q_prd=prd_arr[stn_idx],
        metrics=['ES']
    )[0][0, :, 0, 0]

# benchmark climatology

#  « Climatology benchmark forecast is based on a 40-year climatological
#    sample (1979–2018) of moving 31 d windows of GloFAS-ERA5 river
#    discharge reanalysis values, centred on the date being evaluated
#    (±15 d). From each 1240-valued climatological sample (i.e. 40 years
#    × 31 d window), 11 fixed quantiles (Qn) at 10 % intervals were
#    extracted (Q0, Q10, Q20, …, Q80, Q90, Q100). The fixed quantile
#    climate distribution used therefore varies by lead time, capturing
#    the temporal variability in local river discharge climatology. »
#
#                                      Harrigan et al. (2023), sect. 3.2
#                                 https://doi.org/10.5194/hess-27-1-2023

clm_arr = np.zeros((prd.station.size, prd.step.size, prd.number.size, dts.size))
clm_arr[:] = np.nan

# apply 31-day rolling windows on 40-year climatological sample
wdw = obs.sel(time=slice('1979', '2018')).rolling(time=31, center=True)

# compute quantiles for each day of year over time and window
# (i.e. across 40y and 31d)
qtl = wdw.construct('window').chunk({"time": -1}).groupby('time.dayofyear').quantile(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    dim=('time', 'window')
)

# map climatology benchmark onto observations time steps
for s in range(prd.step.size):
    # get mask selecting time steps where a forecast exists
    msk = np.in1d(dts, prd.valid_time.isel(step=s))
    # use mask to map quantiles in array
    clm_arr[:, s, :, :][:, :, msk] = (
        qtl.sel(dayofyear=prd.valid_time.isel(step=s).dt.dayofyear).values.transpose((0, 2, 1))
    )

es_clm = np.zeros((len(basins), prd.step.size))
for b, basin in enumerate(basins):
    stn_idx = prd.indexes['station'].get_indexer(basins_to_stations[basin])

    es_clm[b] = evalhyd.evalp(
        q_obs=obs.sel(time=dts).values[stn_idx],
        q_prd=clm_arr[stn_idx],
        metrics=['ES']
    )[0][0, :, 0, 0]

ess = 1 - (es_prd / es_clm)

# set transformer object to convert from WGS84 to Lambert93
epsg4326_to_epsg2154 = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:2154", always_xy=True
)

# -----------------------------------------------------------------------------
# plot map for climatology
# -----------------------------------------------------------------------------
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

fig = plt.figure(layout="compressed", figsize=(6, 6))
gs = mpl.gridspec.GridSpec(2, 2, figure=fig)

cmap = mpl.colormaps['RdBu']
norm = mpl.colors.BoundaryNorm(np.arange(-1, 1.1, 0.2), cmap.N)

layout = {
    np.timedelta64(15, 'D').astype('timedelta64[ns]'):
        fig.add_subplot(gs[0, 0], projection=ccrs.epsg(2154)),
    np.timedelta64(20, 'D').astype('timedelta64[ns]'):
        fig.add_subplot(gs[0, 1], projection=ccrs.epsg(2154)),
    np.timedelta64(25, 'D').astype('timedelta64[ns]'):
        fig.add_subplot(gs[1, 0], projection=ccrs.epsg(2154)),
    np.timedelta64(30, 'D').astype('timedelta64[ns]'):
        fig.add_subplot(gs[1, 1], projection=ccrs.epsg(2154))
}

letters = ['a', 'b', 'c', 'd']
letter = 0

for lead, ax in layout.items():
    ldt_days = int(lead / np.timedelta64(1, 'D').astype('timedelta64[ns]'))
    ldt_idx, = prd.indexes['step'].get_indexer([lead],  method="nearest")

    # prepare basemap
    lon_extent = (-5, 12)
    lat_extent = (39, 53.5)

    xc_extent, yc_extent = epsg4326_to_epsg2154.transform(lon_extent, lat_extent)

    ax.set_extent([*xc_extent, *yc_extent], crs=ccrs.epsg(2154))

    ax.add_feature(cfeature.OCEAN, facecolor=cfeature.COLORS['water'])
    ax.add_feature(cfeature.LAND, facecolor="darkgray", alpha=0.5)

    countries = shpreader.Reader(
        shpreader.natural_earth(
            resolution='10m', category='cultural', name='admin_0_countries'
        )
    ).records()
    france = [country for country in countries if country.attributes["NAME_LONG"] == "France"][0]
    shape_feature = cfeature.ShapelyFeature(
        [france.geometry], ccrs.PlateCarree(), facecolor="white"
    )
    ax.add_feature(shape_feature)

    ax.add_feature(cfeature.BORDERS, edgecolor='darkgray', linewidth=0.3)
    ax.add_feature(cfeature.LAKES, facecolor=cfeature.COLORS['water'])
    ax.add_feature(cfeature.RIVERS, edgecolor=cfeature.COLORS['water'], linewidth=0.3, zorder=3)

    ax.set_title(
        r"$\bf{(" + letters[letter] + ")}$"
        + f" {ldt_days} {'day' if ldt_days <= 1 else 'days'}",
        fontsize=mpl.rcParams['xtick.labelsize']
    )
    letter += 1

    # plot results
    for b, basin in enumerate(basins):
        catchment = geopandas.read_file(
            f"data/shapefiles/bassin-{basin.lower()}.shp"
        )
        catchment.crs = "epsg:2154"

        geoplot.polyplot(
            catchment.to_crs(epsg='4326'), ax=ax,
            facecolor=cmap(norm(ess[b, ldt_idx])),
            edgecolor='white', linewidth=0.3,
            projection=geoplot.crs.PlateCarree(),
            zorder=2
        )
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # reset extent because geoplot zoomed on last plotted polygon
    ax.set_extent([*xc_extent, *yc_extent], crs=ccrs.epsg(2154))

    # plot custom made grid
    lon_res = 3
    lat_res = 3
    spacing = 0.25

    lon_vrt_grid_range = np.arange(
        lon_extent[0] - lon_res, lon_extent[1] + lon_res, lon_res
    )
    lat_vrt_grid_range = np.arange(
        lat_extent[0] - lat_res + spacing, lat_extent[1], spacing
    )

    for lon in lon_vrt_grid_range:
        for lat in lat_vrt_grid_range:
            xc, yc = epsg4326_to_epsg2154.transform(lon, lat)

            ax.plot(
                xc, yc,
                marker='o', ms=1.0, markeredgewidth=0, color='darkgray',
                transform=ccrs.epsg(2154),
                zorder=4
            )

    lat_hrz_grid_range = np.arange(
        lat_extent[0] - lat_res, lat_extent[1] + lat_res, lat_res
    )
    lon_hrz_grid_range = np.arange(
        lon_extent[0] - lon_res + spacing, lon_extent[1], spacing
    )

    for lat in lat_hrz_grid_range:
        for lon in lon_hrz_grid_range:
            xc, yc = epsg4326_to_epsg2154.transform(lon, lat)

            ax.plot(
                xc, yc,
                marker='o', ms=1.0, markeredgewidth=0, color='darkgray',
                transform=ccrs.epsg(2154),
                zorder=4
            )

    # use grid lines purely to more easily determine tick locations
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xlocator = mticker.FixedLocator(lon_vrt_grid_range)
    gl.ylocator = mticker.FixedLocator(lat_hrz_grid_range)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': mpl.rcParams['xtick.labelsize']}
    gl.ylabel_style = {'size': mpl.rcParams['ytick.labelsize']}

    # make sure spines are above everything else
    ax.spines['top'].set_zorder(99)
    ax.spines['right'].set_zorder(99)
    ax.spines['bottom'].set_zorder(99)
    ax.spines['left'].set_zorder(99)

# add colorbar
ax = fig.add_subplot(gs[:, :])
pts = ax.scatter(
    x=0, y=0, c=ess[0, 0],
    cmap=cmap, norm=norm
)
ax.set_visible(False)
cbar = fig.colorbar(pts, orientation='horizontal',
                    aspect=40, pad=0.05, extend='min')
cbar.set_ticks(np.arange(-1, 1.1, 1), minor=False)
cbar.set_label("ESS [-]", fontsize=mpl.rcParams['xtick.labelsize'])

plt.rcParams.update(
    {
        "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
        "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
        "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)
    }
)

fig.savefig(f"output/ESS-climatology-map-L93.png", dpi=300)
