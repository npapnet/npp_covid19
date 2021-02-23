#%% read from github.csv
import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

#%% [markdown]
# 
# There are several location for the data
# - [github datasets](https://github.com/datasets/covid-19): long format
# - [John hopkins](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data): wide format

#%% github datasets
def read_total_from_gituhub(url:str=None):
    """ Reads totals for countries from github. 

    Returns:
        DataFrame: containing data for ()
    """    
    url='https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv' if url is None else url
    df = pd.read_csv(url)
    df.drop(['Province/State','Lat', 'Long'], axis=1, inplace=True)
    # df.drop(['Lat', 'Long'], axis=1, inplace=True)
    df.columns = [x if x!='Country/Region' else 'Country' for x in df.columns ]
    return df

def read_aggregated_from_github(url:str=None):
    url='https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'  if url is None else url
    df = pd.read_csv(url)
    return df

def read_worldwide_from_gituhub(url:str=None):
    url='https://raw.githubusercontent.com/datasets/covid-19/master/data/worldwide-aggregate.csv'
    df_sub = pd.read_csv(url)
    df_sub['Date']= pd.to_datetime(df_sub.Date)
    df_sub['dt']= (df_sub['Date']-  df_sub['Date'].values[0])/ datetime.timedelta (days=1)
    df_sub.reset_index(inplace=True, drop=True)
    return df_sub

    
# df_gr.columns 
# df['Country/Region'].unique() 
# [x for x in df['Country/Region'].unique() if 'South' in x]
# %% get data relevant  to as specific country
def get_subset(df, Country='Greece'):
    # get data
    df_sub = df.loc[df.loc[:,'Country']==Country,:].copy()
    # Clean up data
    df_sub.loc[:,'Date']= pd.to_datetime(df_sub['Date'].values)
    df_sub.loc[:,'dt']= (df_sub.loc[:,'Date']-  df_sub.loc[:,'Date'].values[0])/ datetime.timedelta (days=1)
    df_sub.reset_index(inplace=True, drop=True) 
    return df_sub


def remove_zero_entries(df, threshold =1,**kwargs):
    """removes zero entries from dataset
    
    Arguments:
        df {[type]} -- [dataframe to remove entries]
        threshold {float}  -- threshold value
    
    Returns:
        [type] -- [description]
    """    
    f1 = lambda test_list, threshold : next(x for x, val in enumerate(test_list) if val > threshold) # fastest
    # f2 = lambda test_list, threshold : test_list.index(list(filter(lambda i: i > threshold, test_list))[0])
    # f3 = lambda test_list, threshold : list(map(lambda i: i> threshold, test_list)).index(True)
    end_index = None

    end_index = end_index if end_index is not None else df.loc[:,'Confirmed'].count()

    index_value = np.max([0, f1(df.loc[:,'Confirmed'], threshold)-2])
    if kwargs.get('verbose', False):
        print('indexes| start:{}, end:{}'.format(index_value, end_index))
    df_nz = df.loc[list(range(index_value, end_index)),:]
    df_nz.reset_index(inplace=True)
    df_nz.dt = df_nz.dt - df_nz.dt.min()    
    return df_nz

#%%
def convert_data_to_sigmoid_fitting(df,remove_zeros:bool=False):
    dfnew = df.loc[:,['dt', 'Confirmed','Recovered']]
    dfnew.columns = ['Day', 'Confirmed','Recovered']
    dfnew['Day'] = np.int32(dfnew['Day'] )
    if remove_zeros:
        no_zeros = np.max([np.sum(dfnew.Confirmed==0),2])
        dfnew = dfnew.iloc[(no_zeros-2):,:]
        dfnew.reset_index(inplace=True, drop=True)
        dfnew['Day'] = dfnew['Day'] - dfnew['Day'].min()
    return dfnew

#%%=====================================================================
#
#%=====================================================================
if __name__ == "__main__":
    # df = read_total_from_gituhub()
    df = read_aggregated_from_github()
    # dfw = read_worldwide_from_gituhub()
    # dfw_new = convert_data_to_sigmoid_fitting(dfw, remove_zeros=True)
#%% 
    df_gr = get_subset(df)
    # df_gr = get_subset(df, Country='Korea, South')
#%%
    dfnew = convert_data_to_sigmoid_fitting(df_gr, remove_zeros=True)
# %%

    df_gr.plot(x='Date', y='Confirmed')
    df_gr.plot(x='dt', y='Confirmed')

#%%


    df_gr.tail()
# %%
    plt.show()
