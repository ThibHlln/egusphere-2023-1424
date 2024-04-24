# author: Thibault Hallouin, INRAE
# date: June 2023
# description: extract mainland France GloFAS stations from reanalysis data

import xarray as xr
import pandas as pd


# create dataset from all available GRIB files
ds = xr.open_mfdataset(
    'data/raw/GloFAS-ERA5_v2.1_river_discharge_reanalysis_*.nc'
)

# subset dataset for each station based on its lat/lon and save to NetCDF file
meta = pd.read_csv('data/Harrigan2023-GloFAS-river_points.csv')

for i in range(len(meta)):
    stn, lat, lon = meta.loc[:, ['GloFAS_ID', 'LAT_GloFAS', 'LON_GloFAS']].iloc[i]

    ds.sel(latitude=lat, longitude=lon, method='nearest').to_netcdf(
        f'data/modified/GloFAS-ERA5_v2.1_river_discharge_reanalysis_{stn}.nc'
    )
