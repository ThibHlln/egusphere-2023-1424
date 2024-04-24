# author: Thibault Hallouin, INRAE
# date: June 2023
# description: extract mainland France GloFAS stations from reforecast data

import xarray as xr
import pandas as pd


# create dataset from all available GRIB files
ds_cf = xr.open_mfdataset(
    'data/raw/GloFAS-v2.2_river_discharge_reforecast_*_cf.grib',
    engine='cfgrib'
).expand_dims('number')

ds_pf = xr.open_mfdataset(
    'data/raw/GloFAS-v2.2_river_discharge_reforecast_*_pf.grib',
    engine='cfgrib'
)

ds = xr.combine_by_coords([ds_cf, ds_pf], combine_attrs='override')

# subset dataset for each station based on its lat/lon and save in NetCDF file
meta = pd.read_csv('data/Harrigan2023-GloFAS-river_points.csv')

for i in range(len(meta)):
    stn, lat, lon = meta.loc[:, ['GloFAS_ID', 'LAT_GloFAS', 'LON_GloFAS']].iloc[i]

    ds.sel(latitude=lat, longitude=lon, method='nearest').to_netcdf(
        f'data/modified/GloFAS-v2.2_river_discharge_reforecast_{stn}.nc'
    )
