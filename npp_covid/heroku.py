import requests
import numpy as np
import pandas as pd

#%%
def hag_get_tests():
    ''' download tests data 
    
    USes the https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests to 

    other potential sources
    # https://ourworldindata.org/coronavirus-testing
 
    # https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests

    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/total-tests")
    dictr = response.json()
    df = pd.json_normalize (dictr['total_tests'])
    df['tests']= pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    return df

def ha_get_all():
    ''' download tests data 
    
    USes the https://covid-19-greece.herokuapp.com/all

    other potential sources
    # https://ourworldindata.org/coronavirus-testing
 
    # https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests

    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/all")
    dictr = response.json()
    df = pd.json_normalize (dictr['cases'])
    # df['tests']= pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    return df


def hag_create_df():
    df_cases =  ha_get_all()
    df_tests= hag_get_tests()

    df = df_cases.set_index('date').join(df_tests.set_index('date'))
    df['rapid-tests']=pd.to_numeric(df['rapid-tests'].fillna(0), downcast='integer')
    df['tests']=pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    return df


def hag_get_intensive_care():
    ''' downloads history of intensive-care 
    
    USes the https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/ to 

    other potential sources
    # https://ourworldindata.org/coronavirus-testing
 
    # https://covid-19-greece.herokuapp.com/docs/#/Daily%20recorded%20events/get_total_tests

    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/intensive-care")
    dictr = response.json()
    df = pd.json_normalize (dictr['cases'])
    df['intensive_care']= pd.to_numeric(df['intensive_care'].fillna(0), downcast='integer')
    return df



def prepare_cases(new_cases , 
    cutoff=25, 
    periods_count=14, 
    stds=4):
    
    smoothed = new_cases.rolling(periods_count,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=stds).round()
    
    idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return  smoothed

def hag_get_vaccinations():
    ''' download vaccination data data 
    
    Uses the https://covid-19-greece.herokuapp.com/docs/#

    The data come in regions, so this function first groups by date the data to provide the total number of vaccinations
    There are alos other data available
    '''
    response = requests.get("https://covid-19-greece.herokuapp.com/vaccinations-per-region-history")
    dictr = response.json()
    df = pd.json_normalize (dictr['vaccinations-history'])
    # df['tests']= pd.to_numeric(df['tests'].fillna(0), downcast='integer')
    
    # group by date to provide cumulative results. 
    dn = df.drop(columns = ['area_en', 'area_gr', 'dailydose1', 'dailydose2', 'daydiff', 'daytotal']) 
    # 
    dfsum = dn.groupby('referencedate').sum()

    return dfsum