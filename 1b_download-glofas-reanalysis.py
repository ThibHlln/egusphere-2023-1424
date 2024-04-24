# author: Thibault Hallouin, INRAE
# date: June 2023
# description: download GloFAS reanalysis data using CDS API

import cdsapi
import zipfile
import os

c = cdsapi.Client()

folder = 'data/raw'

for year in range(1979, 2019):

    filename = (
        f'GloFAS-ERA5_v2.1_river_discharge_reanalysis_{year}'
    )

    # download reanalysis data
    c.retrieve(
        'cems-glofas-historical',
        {
            'system_version': 'version_2_1',
            'variable': 'river_discharge_in_the_last_24_hours',
            'format': 'netcdf4.zip',
            'hydrological_model': 'htessel_lisflood',
            'product_type': 'consolidated',
            'hmonth': [
                'april', 'august', 'december',
                'february', 'january', 'july',
                'june', 'march', 'may',
                'november', 'october', 'september',
            ],
            'hyear': year,
            'hday': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'area': [
                51.5, -5, 41, 10,
            ],
        },
        f'{folder}/{filename}.zip'
    )

    # unzip archive
    with zipfile.ZipFile(f'{folder}/{filename}.zip', 'r') as zip_file:
        zip_file.extractall(folder)

    # delete archive
    os.remove(f'{folder}/{filename}.zip')

    # rename netCDF file
    os.rename(f'{folder}/data.nc', f'{folder}/{filename}.nc')
