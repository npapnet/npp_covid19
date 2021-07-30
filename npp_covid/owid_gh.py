#%%[markdown]
# From [Owid](https://github.com/owid/covid-19-data) 
# 
# Eg:
# - vaccination are in https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv

#%%
#%% read from github.csv
import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
#%% CONSTANTS
LOCAL_OWID_FILE = 'data/owid_covid_data.csv'
# %%

def get_owid_data_file(url = None, save_locally=False):
    url='https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv' if url is None else url
    df = pd.read_csv(url)
    if save_locally:
        df.to_csv(LOCAL_OWID_FILE)
    return df
# %%
if __name__=='__main__':
    #get_owid_data_file(save_locally=True)

    df = pd.read_csv(LOCAL_OWID_FILE)
    # %%
    df.iso_code.unique()
    # %%
    dfg = df[df.iso_code == 'GRC']
    # %%
    df.columns
    # %%
    dfg.loc[:,['reproduction_rate', 'icu_patients',
        'icu_patients_per_million', 'hosp_patients',
        'hosp_patients_per_million', 'weekly_icu_admissions',
        'weekly_icu_admissions_per_million', 'weekly_hosp_admissions',
        'weekly_hosp_admissions_per_million', 'new_tests', 'total_tests',
        'total_tests_per_thousand', 'new_tests_per_thousand']]
    # %%
    res = np.sum(np.logical_not(dfg.isna()))
    for i in res.index:
        print("{} : {}".format(i, res[i]))
    # %%
