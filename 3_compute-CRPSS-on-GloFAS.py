# author: Thibault Hallouin, INRAE
# date: June 2023
# description: compute CRPSS against persistence and climatology benchmarks using evalhyd

import xarray as xr
import numpy as np
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

# evaluate
# forecast
crps_forecast, = evalhyd.evalp(
    q_obs=obs.sel(time=dts).values,
    q_prd=prd_arr,
    metrics=['CRPS_FROM_ECDF'],
    q_thr=thr,
    events='low'
)

# benchmark persistence

#  « Persistence benchmark forecast is defined as the single GloFAS-ERA5
#    daily river discharge of the day preceding the reforecast start
#    date. The same river discharge value is used for all lead times.
#    For example, for a forecast issued on 3 January at 00:00 UTC, the
#    persistence benchmark forecast is the average river discharge over
#    the 24 h time step from 2 January 00:00 UTC to 3 January 00:00 UTC,
#    and the same value is used as benchmark for all 30 lead times
#    (i.e., 4 January to 2 February). »
#
#                                      Harrigan et al. (2023), sect. 3.2
#                                 https://doi.org/10.5194/hess-27-1-2023
#
prs_arr = np.zeros(
    (prd.station.size, prd.step.size, dts.size)
)
prs_arr[:] = np.nan

# map persistence benchmark onto observations time steps
for s in range(prd.step.size):
    # get mask selecting time steps where a forecast exists
    msk = np.in1d(dts, prd.valid_time.isel(step=s))
    # use mask to map observations in array
    prs_arr[:, s, :][:, msk] = obs.sel(time=prd.time).values

# compute persistence benchmark CRPS one site at a time
# because evalhyd deterministic evaluation is 2D-only
crps_persistence = np.zeros((prd.station.size, prd.step.size, 1, 1))
crps_persistence[:] = np.nan

for s in range(prd.station.size):
    crps_persistence[s] = evalhyd.evald(
        q_obs=obs.isel(station=s).sel(time=dts).values[np.newaxis, :],
        q_prd=prs_arr[s].copy(),
        metrics=['MAE']
    )[0]

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
#
clm_arr = np.zeros(
    (prd.station.size, prd.step.size, prd.number.size, dts.size)
)
clm_arr[:] = np.nan

# apply 31-day rolling windows on 40-year climatological sample
wdw = obs.sel(time=slice('1979', '2018')).rolling(time=31, center=True)

# compute quantiles for each day of year over time and window
# (i.e. across 40y and 31d)
qtl = (
    wdw.construct('window').chunk({"time": -1}).groupby('time.dayofyear')
    .quantile(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        dim=('time', 'window')
    )
)

# map climatology benchmark onto observations time steps
for s in range(prd.step.size):
    # get mask selecting time steps where a forecast exists
    msk = np.in1d(dts, prd.valid_time.isel(step=s))
    # use mask to map quantiles in array
    clm_arr[:, s, :, :][:, :, msk] = (
        qtl.sel(dayofyear=prd.valid_time.isel(step=s).dt.dayofyear)
        .values.transpose((0, 2, 1))
    )

crps_climatology = evalhyd.evalp(
    q_obs=obs.sel(time=dts).values,
    q_prd=clm_arr,
    metrics=['CRPS_FROM_ECDF']
)[0]

# compute skill scores
crpss_persistence = 1 - (crps_forecast / crps_persistence)
crpss_climatology = 1 - (crps_forecast / crps_climatology)

np.savetxt(
    "output/CRPSS-persistence.csv",
    np.squeeze(crpss_persistence),
    delimiter=','
)

np.savetxt(
    "output/CRPSS-climatology.csv",
    np.squeeze(crpss_climatology),
    delimiter=','
)
