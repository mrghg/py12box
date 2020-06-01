#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:48:20 2018

@author: chxmr
"""

import matplotlib.pyplot as plt
import xarray as xr
import json
from os.path import join
from bisect import bisect
from acrg_config.paths import paths
import pandas as pd
import numpy as np
import glob
from acrg_grid import areagrid
import cpickle as pickle

acrg_path = paths.acrg

with open(join(acrg_path, "acrg_site_info.json")) as f:
    site_info=json.load(f)

files = sorted(glob.glob("/home/chxmr/work/py12box_parameters/co2_lifetime.mz4.h0.*.nc"))

# Get MOZART grid
with xr.open_dataset(files[0], decode_times=False) as ds:
    lon = ds["lon"]
    lat = ds["lat"]
    lev = ds["lev"]
    ilev = ds["ilev"]

# Calculate weight by cos-lat and pressure for each level
lat_weights = np.cos(lat* np.pi / 180.)
lev_weights = ilev.values[1:] - ilev.values[:-1]
weights = np.zeros((len(lev), len(lat)))
for la in range(len(lat)):
    for le in range(len(lev)):
        weights[le, la] = lat_weights[la]*lev_weights[le]

area = areagrid(lat.values, lon.values)

# Box definitions
box_latitudes = {0: [30,90],
                 1: [0,30],
                 2: [-30,0],
                 3: [-90,-30]}

box_levels = {0: [1000., 500.],
              1: [500., 200.],
              2: [200., 0.]}

mf_box = np.zeros((12, 12)) #(box, month)
q_box = np.zeros((4, 12))

# Find location of AGAGE stations
stations = {"MHD": {"box":0},
            "THD": {"box":0},
            "RPB": {"box":1},
            "SMO": {"box":2},
            "CGO": {"box":3}}
            
for s in stations:

    # Find nearest grid cell
    if site_info[s]["longitude"] < 0.:
        site_lon = 360. + site_info[s]["longitude"]
    else:
        site_lon = site_info[s]["longitude"]
    loni = bisect(lon, site_lon)
    stations[s]["loni"] = loni

    lati = bisect(lat, site_info[s]["latitude"])
    stations[s]["lati"] = lati

    # empty array to store monthly means
    stations[s]["monthly_mean"] = np.zeros(12)

    # list to store mole fractions
    stations[s]["mf"] = []
    
    # list to store pollution
    stations[s]["mf_5d"] = []

    # list to store pollution
    stations[s]["time_wh"] = []

    # list to store pollution
    stations[s]["mf_wh"] = []

    # list to store pollution
    stations[s]["time"] = []


threshold = 1e-7

for fi, f in enumerate(files):

    print("...reading %s" %f)
    
    with xr.open_dataset(f, decode_times=False) as ds:
        mf = ds["CO2_inf_VMR_avrg"]
        mf_5d = ds["CO2_5d_VMR_avrg"]
        q = ds["CO2_inf_SRF_EMIS_avrg"]
        date = ds["date"]
        datesec = ds["datesec"]
    
    time = pd.to_datetime(["%s %02i:00" % (d, s/3600) for d, s in zip(date.values, datesec.values)])
    
    # store station-specific info
    for s in stations:
        mfs_5d = mf_5d[:, -3, stations[s]["lati"], stations[s]["loni"]]
        mfs = mf[:, -3, stations[s]["lati"], stations[s]["loni"]]
        stations[s]["time"].append(time)
        stations[s]["mf"].append(mfs)
        stations[s]["mf_5d"].append(mfs_5d)

        wh = np.where(mfs_5d.values < threshold)
        if len(wh[0]) > 0:
            stations[s]["mf_wh"].append(mfs[wh])
            stations[s]["time_wh"].append(time[wh])
            stations[s]["monthly_mean"][fi] = np.mean(mfs[wh])
        else:
            stations[s]["monthly_mean"][fi] = np.nan
    
    # average over longitude and time
    mf_av = np.mean(mf.values, axis = (0, 3))
    q_av = np.mean(q.values, axis = 0)
    
    # store higher box model levels and surface emissions
    for lat_box in range(4):
        wh_lat = np.where((lat > box_latitudes[lat_box][0]) * (lat < box_latitudes[lat_box][1]))

        # Store surface emissions
        q_box[lat_box, fi] = np.sum(q_av[wh_lat[0], :]*area[wh_lat[0], :])*365.25*24.*3600. # convert to kg/yr

        # Store higher level box mole fractions
        for lev_box in range(1, 3, 1):
            wh_lev = np.where((lev < box_levels[lev_box][0]) * (lev > box_levels[lev_box][1]))
            mf_box[lev_box*4 + lat_box, fi] = np.sum(mf_av[np.ix_(wh_lev[0], wh_lat[0])] * \
                                                       weights[np.ix_(wh_lev[0], wh_lat[0])]) / \
                                              np.sum(weights[np.ix_(wh_lev[0], wh_lat[0])])


for surface_box in range(4):
    for month in range(12):
        box_count = 0
        for s in stations:
            if stations[s]["box"] == surface_box:
                if np.isfinite(stations[s]["monthly_mean"][month]):
                    mf_box[surface_box, month] += stations[s]["monthly_mean"][month]
                    box_count += 1
        mf_box[surface_box, month] /= box_count


with open("/home/chxmr/work/py12box_parameters/co2_boxed_up.p", "wb") as f:
    pickle.dump({"mf": mf_box, "q": q_box}, f)



