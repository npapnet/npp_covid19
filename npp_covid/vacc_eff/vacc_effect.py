'''module that contains functions for vaccination effect

'''
import matplotlib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import npp_covid.heroku as io_cd

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

VACC_PERC_STR = 'vacc_perc'

def prepare_dataframe(df0, country_pop =1e7):
    """Prepares a dataframe for the vaccination effect with the data

    operations
    - fills vaccination data with nearest 
    - fills na values in vaccination data with zero
    - creates conf, that contains a smoothed version of confirmed. 

    Args:
        df0 ([type]): dataframe with columns ['confirmed',	'deaths',	'intensive_care', 'totaldistinctpersons']
        country_pop ([type], optional): [description]. Defaults to GREECE_POPULATION.

    Returns:
        df_prep: [description]
    """    
    tmp = df0['totaldistinctpersons'].reset_index().interpolate(method='nearest')
    tmp.fillna(0,inplace=True)
    tmp.set_index('date', inplace=True)

    df_prep = df0.copy()
    df_prep['totaldistinctpersons'] = tmp['totaldistinctpersons']
    df_prep[VACC_PERC_STR] = tmp['totaldistinctpersons']/country_pop
    # smooth confirmed
    df_prep['conf']= df0['confirmed'].rolling(window=15, center=True).mean()
    df_prep.dropna(inplace=True)

    return df_prep

def plot_deaths_conf_wrt_vaccRate(df0,   n_shift =20, figsize=(12,8)):
    """fucntion that plots the data 
    - x: confirmed cases
    - y: deaths shifted /confirmed ratio
    - markers and hue depict: 'Vaccination percentage

    Args:
        df0 ([type]): pd.DataFrame
        shift (int, optional): [description]. Defaults to 20.
        figsize (tuple, optional): [description]. Defaults to (12,8).
    """    
    df0['markers']=pd.cut(df0['totaldistinctpersons'], bins=5)
    df0['deaths_shift'] = df0['deaths'].shift(-n_shift )
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df0, x='conf', y='deaths_shift', style="markers", hue='markers')

