# author: Thibault Hallouin, INRAE
# date: June 2023
# description: plot CRPSS against persistence and climatology benchmarks on map

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


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

# set transformer object to convert from WGS84 to Lambert93
epsg4326_to_epsg2154 = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:2154", always_xy=True
)

# -----------------------------------------------------------------------------
# plot comparison maps for persistence and climatology
# -----------------------------------------------------------------------------
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

for benchmark_name, benchmark_data, benchmark_leadtimes in zip(
        ('persistence', 'climatology'),
        (hallouin_persistence, hallouin_climatology),
        ([1, 3, 5, 10], [15, 20, 25, 30])
):
    fig = plt.figure(layout="compressed", figsize=(6, 6))
    gs = mpl.gridspec.GridSpec(2, 2, figure=fig)

    cmap = mpl.colormaps['RdBu']
    norm = mpl.colors.BoundaryNorm(np.arange(-1, 1.1, 0.2), cmap.N)

    layout = {
        np.timedelta64(benchmark_leadtimes[0], 'D').astype('timedelta64[ns]'):
            fig.add_subplot(gs[0, 0], projection=ccrs.epsg(2154)),
        np.timedelta64(benchmark_leadtimes[1], 'D').astype('timedelta64[ns]'):
            fig.add_subplot(gs[0, 1], projection=ccrs.epsg(2154)),
        np.timedelta64(benchmark_leadtimes[2], 'D').astype('timedelta64[ns]'):
            fig.add_subplot(gs[1, 0], projection=ccrs.epsg(2154)),
        np.timedelta64(benchmark_leadtimes[3], 'D').astype('timedelta64[ns]'):
            fig.add_subplot(gs[1, 1], projection=ccrs.epsg(2154))
    }

    letters = ['a', 'b', 'c', 'd']
    letter = 0

    for lead, ax in layout.items():
        ldt_days = int(lead / np.timedelta64(1, 'D').astype('timedelta64[ns]'))
        ldt_idx = prd.indexes['step'].get_indexer([lead],  method="nearest")

        # prepare basemap
        lon_extent = (-5, 12)
        lat_extent = (39, 53.5)

        xc_extent, yc_extent = epsg4326_to_epsg2154.transform(lon_extent, lat_extent)

        ax.set_extent([*xc_extent, *yc_extent], crs=ccrs.epsg(2154))

        ax.add_feature(cfeature.OCEAN, facecolor=cfeature.COLORS['water'])
        ax.add_feature(cfeature.LAND, facecolor="darkgray", alpha=0.5)
        # ax.add_feature(cfeature.COASTLINE)

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
        ax.add_feature(cfeature.RIVERS, edgecolor=cfeature.COLORS['water'], linewidth=0.3)

        ax.set_title(r"$\bf{(" + letters[letter] + ")}$" + f" {ldt_days} {'day' if ldt_days <= 1 else 'days'}",
                     fontsize=mpl.rcParams['xtick.labelsize'])
        letter += 1

        # retrieve locations
        lat, lon, lat_, lon_ = (
            meta.loc[:, ['LAT_Provided', 'LON_Provided', 'LAT_GloFAS', 'LON_GloFAS']].values.T
        )
        lat[np.isnan(lat)] = lat_[np.isnan(lat)]
        lon[np.isnan(lon)] = lon_[np.isnan(lon)]

        xc, yc = epsg4326_to_epsg2154.transform(lon, lat)

        # plot results
        pts = ax.scatter(
            x=xc, y=yc, c=benchmark_data[:, ldt_idx], s=21,
            edgecolors='white', linewidths=0.2,
            cmap=cmap, norm=norm,
            zorder=20
        )

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
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        ax.spines['top'].set_zorder(99)
        ax.spines['right'].set_zorder(99)
        ax.spines['bottom'].set_zorder(99)
        ax.spines['left'].set_zorder(99)

    # add colorbar
    ax = fig.add_subplot(gs[:, :])
    pts = ax.scatter(
        x=0, y=0, c=benchmark_data[0, 0],
        cmap=cmap, norm=norm
    )
    ax.set_visible(False)
    cbar = fig.colorbar(pts, orientation='horizontal',
                        aspect=40, pad=0.05, extend='min')
    cbar.set_ticks(np.arange(-1, 1.1, 1), minor=False)
    cbar.set_label("CRPSS [-]", fontsize=mpl.rcParams['xtick.labelsize'])

    plt.rcParams.update(
        {
            "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
            "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
            "savefig.facecolor": (1.0, 1.0, 1.0, 0.0)
        }
    )

    fig.savefig(f"output/CRPSS-{benchmark_name}-map-L93.png", dpi=300)
