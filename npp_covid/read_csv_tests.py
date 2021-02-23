#%% read from github.csv
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_tests_aggregated_from_github():
    url='https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/testing/covid-testing-all-observations.csv'
    usef= lambda x: not (x in [4,5,6])
    df = pd.read_csv(url, usecols=usef)
    return df

def get_testing_subset(df, Country='GRC'):
    # get data
    df_sub = df.loc[df.loc[:,'ISO code']==Country,:].copy()
    df_sub = df_sub.drop(labels=['Entity', 'Source URL', 'Source label', 'Notes'], axis =1)
    return df_sub
#%%=====================================================================
#
#%=====================================================================
if __name__ == "__main__":
    # df = read_total_from_gituhub()
    df = read_tests_aggregated_from_github()
    # dfw = read_worldwide_from_gituhub()
    # dfw_new = convert_data_to_sigmoid_fitting(dfw, remove_zeros=True)
#%%
    df.tail()
#%% 
    df_gr = get_testing_subset(df)
    df_gr.tail()
    df_gr['7-day smoothed daily change'].plot()
#%%
    dfnew = convert_data_to_sigmoid_fitting(df_gr, remove_zeros=True)
# %%

    df_gr.plot(x='Date', y='Confirmed')
    df_gr.plot(x='dt', y='Confirmed')

#%%


    df_gr.tail()
# %%
    plt.show()
