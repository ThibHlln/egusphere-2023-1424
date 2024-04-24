# author: Thibault Hallouin, INRAE
# date: June 2023
# description: download GloFAS reforecast data using CDS API

import cdsapi

c = cdsapi.Client()

products = [
    'control_reforecast',
    'ensemble_perturbed_reforecasts'
]

years = [f'{y:4d}' for y in range(1999, 2019)]

months = {
    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12',
}

leadtimes = ['24', '48', '72', '96', '120', '144', '168', '192', '216',
             '240', '288', '336', '384', '432', '480', '600', '720']

for product in products:
    suffix = 'cf' if product == 'control_reforecast' else 'pf'
    for yyyy in years:
        for month, mm in months.items():
            c.retrieve(
                'cems-glofas-reforecast',
                {
                    'variable': 'river_discharge_in_the_last_24_hours',
                    'format': 'grib',
                    'system_version': 'version_2_2',
                    'hydrological_model': 'htessel_lisflood',
                    'product_type': product,
                    'hyear': yyyy,
                    'hmonth': month,
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
                    'leadtime_hour': leadtimes,
                    'area': [51.5, -5, 41, 10],
                },
                f'data/raw/GloFAS-v2.2_river_discharge_reforecast_'
                f'{yyyy}{mm}_{suffix}.grib'
            )
